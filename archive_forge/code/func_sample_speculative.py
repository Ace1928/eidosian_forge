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
def sample_speculative(logits, logits_draft, tokens_draft, top_k=1, top_p=0.0, temperature=1.0):
    """Algorithm 1 from [1]
    [1] Fast Inference from Transformers via Speculative Decoding
    Yaniv Leviathan, Matan Kalman, Yossi Matias
    https://arxiv.org/abs/2211.17192

    Arguments:
        logits: Tensor of shape (batch_size, seqlen + 1, vocab_size)
        logits_draft: Tensor of shape (batch_size, seqlen, vocab_size)
        tokens_draft: Tensor of shape (batch_size, seqlen)
    Return:
        tokens: Tensor of shape (batch_size, seqlen + 1)
        num_generated_tokens: Tensor of shape (batch_size), with value in [1, seqlen + 1].
            For each sequence in the batch, the number of valid tokens that were sampled by
            speculative sampling.
    """
    batch, seqlen_p_1, vocab_size = logits.shape
    seqlen = seqlen_p_1 - 1
    assert logits_draft.shape == (batch, seqlen, vocab_size)
    assert tokens_draft.shape == (batch, seqlen)
    assert tokens_draft.dtype in [torch.int64, torch.int32]
    if top_p > 0.0:
        assert top_p <= 1.0, 'top-p should be in (0, 1].'
    logits = logits / temperature if temperature != 1.0 else logits.clone()
    logits_draft = logits_draft / temperature if temperature != 1.0 else logits_draft.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        modify_logits_for_top_k_filtering(logits, top_k)
        modify_logits_for_top_k_filtering(logits_draft, top_k)
    modify_logits_for_top_p_filtering(logits, top_p)
    modify_logits_for_top_p_filtering(logits_draft, top_p)
    probs = torch.softmax(logits, dim=-1)
    probs_draft = torch.softmax(logits_draft, dim=-1)
    gather = lambda probs, tokens: rearrange(probs.gather(dim=-1, index=rearrange(tokens, '... -> ... 1')), '... 1 -> ...')
    accepted = torch.rand(batch, seqlen, device=probs.device) * gather(probs_draft, tokens_draft) <= gather(probs[:, :-1], tokens_draft)
    accepted_all = accepted.all(dim=-1)
    first_rejected_idx = torch.where(accepted_all, seqlen, accepted.int().argmin(dim=-1))
    probs_diff = torch.clamp(probs[:, :-1] - probs_draft, min=0.0)
    resample_probs = torch.cat([probs_diff, probs[:, -1:]], dim=1)
    resample_probs = rearrange(resample_probs.gather(dim=1, index=repeat(first_rejected_idx, 'b -> b 1 d', d=vocab_size)), 'b 1 d -> b d')
    resample = torch.multinomial(resample_probs, num_samples=1).squeeze(dim=-1)
    tokens = F.pad(tokens_draft, (0, 1))
    tokens[:, first_rejected_idx] = resample
    return (tokens, first_rejected_idx + 1)