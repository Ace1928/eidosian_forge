import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
@torch.inference_mode()
def decode_speculative(input_ids, model, model_draft, max_length, speculative_lookahead=3, top_k=1, top_p=0.0, temperature=1.0, eos_token_id=None, vocab_size=None, tensor_parallel=1, cg=False, enable_timing=False, debug=False):
    """
    TD: WIP, for my own understanding, lightly tested. Only support batch_size == 1 for now.

    Speculative decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    assert batch_size == 1, 'Speculative decoding implementation only supports batch_size=1'
    assert eos_token_id is None, "Speculative decoding implementation doesn't support eos_token_id"
    if cg:
        if not hasattr(model_draft, '_decoding_cache'):
            model_draft._decoding_cache = None
        model_draft._decoding_cache = update_graph_cache(model_draft, model_draft._decoding_cache, batch_size, seqlen_og, max_length, decoding_seqlens=(1, 2), tensor_parallel=tensor_parallel)
        inference_params_draft = model_draft._decoding_cache.inference_params
        inference_params_draft.reset(max_length, batch_size)
        if not hasattr(model, '_decoding_cache'):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(model, model._decoding_cache, batch_size, seqlen_og, max_length, decoding_seqlens=range(1, speculative_lookahead + 2), tensor_parallel=tensor_parallel)
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params_draft = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params, model, num_last_tokens=1, cg=False):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            seqlen = input_ids.shape[1]
            if True:
                cache_seqlens = torch.full((input_ids.shape[0],), inference_params.seqlen_offset, dtype=torch.int32, device=input_ids.device)
            else:
                cache_seqlens = inference_params.lengths_per_sample
            position_ids = cache_seqlens[:, None] + torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(input_ids, position_ids=position_ids, inference_params=inference_params, num_last_tokens=num_last_tokens).logits
        else:
            assert num_last_tokens <= input_ids.shape[1]
            logits = model._decoding_cache.run(input_ids, position_ids, inference_params.seqlen_offset)[:, -num_last_tokens:]
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(input_ids, get_logits_fn, inference_params, sample_fn, num_tokens=1):
        """Sample `num_tokens` tokens from the model, given the previous logits.
        Also return the logits of the sampled tokens.
        Arguments:
            input_ids: (batch, seqlen)
        Return:
            tokens: (batch, num_tokens)
            scores: (batch, num_tokens), which contains @previous_logits and the logits of the next
                (num_tokens - 1) tokens. The logits of the last token isn't computed.
        """
        assert num_tokens >= 1
        sequences, scores = ([input_ids], [])
        for i in range(num_tokens):
            scores.append(get_logits_fn(sequences[-1], inference_params)[:, -1])
            inference_params.seqlen_offset += sequences[-1].shape[1]
            sequences.append(sample_fn(scores[-1]).unsqueeze(1))
        return (torch.cat(sequences[1:], dim=1), torch.stack(scores, dim=1))
    sampling_kwargs = dict(top_k=top_k, top_p=top_p, temperature=temperature)
    sample_fn = partial(sample, **sampling_kwargs)
    get_logits_main = partial(get_logits, model=model, cg=cg)
    get_logits_draft = partial(get_logits, model=model_draft, cg=cg)
    sample_tokens_main = partial(sample_tokens, get_logits_fn=get_logits_main, sample_fn=sample_fn, inference_params=inference_params)
    sample_tokens_draft = partial(sample_tokens, get_logits_fn=get_logits_draft, sample_fn=sample_fn, inference_params=inference_params_draft)
    if debug:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        start = time.time()
    sequences, scores = ([input_ids], [])
    num_main_model_calls = 0
    num_draft_tokens = 0
    num_accepted_tokens_history = []
    if seqlen_og >= max_length - 1:
        tokens, scores_new = sample_tokens_main(input_ids, num_tokens=1)
        sequences.append(tokens)
        scores.append(scores_new)
    else:
        n_spec_tokens = min(speculative_lookahead, max_length - seqlen_og - 1)
        tokens_draft, scores_draft = sample_tokens_draft(input_ids, num_tokens=n_spec_tokens)
        num_draft_tokens += n_spec_tokens
        if debug:
            scores_draft_ref = model_draft(torch.cat([input_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1).logits
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())
        logits = get_logits_main(torch.cat([input_ids, tokens_draft], dim=1), inference_params, num_last_tokens=n_spec_tokens + 1)
        num_main_model_calls += 1
        if debug:
            logits_ref = model(torch.cat([input_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1).logits
            print((logits - logits_ref).abs().max())
        tokens, num_generated_tokens = sample_speculative(logits, scores_draft, tokens_draft, **sampling_kwargs)
        num_accepted_tokens_history.append(num_generated_tokens - 1)
        if debug:
            print(tokens)
            print(num_generated_tokens)
        sequences.append(tokens[:1, :num_generated_tokens[0]])
        scores.append(logits[:1, :num_generated_tokens[0]])
        num_generated = num_generated_tokens[0].item()
        inference_params.seqlen_offset = seqlen_og + num_generated - 1
        inference_params_draft.seqlen_offset = inference_params.seqlen_offset - 1 if num_generated > 1 else inference_params.seqlen_offset
        if debug:
            cur_ids = torch.cat([input_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
    while True:
        if inference_params.seqlen_offset >= max_length - 1:
            break
        if inference_params.seqlen_offset >= max_length - 2:
            tokens, scores_new = sample_tokens_main(sequences[-1][:, -1:], num_tokens=1)
            sequences.append(tokens)
            scores.append(scores_new)
            break
        n_spec_tokens = min(speculative_lookahead, max_length - inference_params_draft.seqlen_offset - 2)
        tokens_draft, scores_draft = sample_tokens_draft(sequences[-1][:, -2:], num_tokens=n_spec_tokens)
        num_draft_tokens += n_spec_tokens
        if debug:
            scores_draft_ref = model_draft(torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1).logits
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())
        logits = get_logits_main(torch.cat([sequences[-1][:, -1:], tokens_draft], dim=1), inference_params, num_last_tokens=n_spec_tokens + 1)
        num_main_model_calls += 1
        if debug:
            logits_ref = model(torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1).logits
            print((logits - logits_ref).abs().max())
        tokens, num_generated_tokens = sample_speculative(logits, scores_draft, tokens_draft, **sampling_kwargs)
        num_accepted_tokens_history.append(num_generated_tokens - 1)
        if debug:
            print(tokens)
            print(num_generated_tokens)
        sequences.append(tokens[:1, :num_generated_tokens[0]])
        scores.append(logits[:1, :num_generated_tokens[0]])
        num_generated = num_generated_tokens[0].item()
        inference_params.seqlen_offset += num_generated
        inference_params_draft.seqlen_offset = inference_params.seqlen_offset - 1 if num_generated > 1 else inference_params.seqlen_offset
        if debug:
            cur_ids = torch.cat([cur_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f'Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms')
        print(f'Number of calls to main model: {num_main_model_calls}')
        print(f'Acceptance rate: {torch.cat(num_accepted_tokens_history).sum().item() / num_draft_tokens * 100:.2f}%')
    sequences = torch.cat(sequences, dim=1)
    scores = torch.cat(scores, dim=1)
    if debug:
        scores_ref = model(sequences).logits
        print((scores - scores_ref[:, seqlen_og - 1:-1]).abs().max())
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=sequences, scores=scores)