import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def megatron_generate(self, inputs, attention_mask=None, max_length=None, max_new_tokens=None, num_beams=None, temperature=None, top_k=None, top_p=None, length_penalty=None, **kwargs):
    """
        Generate method for GPT2 model. This method is used for inference. Supports both greedy and beam search along
        with sampling. Refer the Megatron-LM repo for more details

        Args:
            inputs (torch.Tensor): input ids
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            max_length (int, optional): max length of the generated sequence. Defaults to None.
            Either this or max_new_tokens should be provided.
            max_new_tokens (int, optional): max number of tokens to be generated. Defaults to None.
            Either this or max_length should be provided.
            num_beams (int, optional): number of beams to use for beam search. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top k tokens to consider for sampling. Defaults to 0.0.
            top_p (float, optional): tokens in top p probability are considered for sampling. Defaults to 0.0.
            length_penalty (float, optional): length penalty for beam search. Defaults to None.
            kwargs: additional key-value arguments
        """
    args = get_args()
    if args.model_type_name != 'gpt':
        raise NotImplementedError('Generate method is not implemented for this model')
    if args.data_parallel_size > 1:
        raise ValueError('Generate method requires data parallelism to be 1')
    if args.sequence_parallel:
        raise ValueError('Generate method requires sequence parallelism to be False')
    if args.recompute_granularity is not None:
        raise ValueError('Checkpoint activations cannot be set for inference')
    if args.vocab_file is None:
        raise ValueError('Vocab file is required for inference')
    if max_length is None and max_new_tokens is None:
        raise ValueError('`max_length` or `max_new_tokens` are required for inference')
    if temperature is None:
        temperature = 1.0
    elif not 0.0 < temperature <= 100.0:
        raise ValueError('temperature must be a positive number less than or equal to 100.0')
    if top_k is None:
        top_k = 0
    elif not 0 <= top_k <= 1000:
        raise ValueError('top_k must be a positive number less than or equal to 1000')
    if top_p is None:
        top_p = 0.0
    elif top_p > 0.0 and top_k > 0.0:
        raise ValueError('top_p and top_k sampling cannot be set together')
    elif not 0.0 <= top_p <= 1.0:
        raise ValueError('top_p must be less than or equal to 1.0')
    top_p_decay = kwargs.get('top_p_decay', 0.0)
    if not 0.0 <= top_p_decay <= 1.0:
        raise ValueError('top_p_decay must be less than or equal to 1.0')
    top_p_bound = kwargs.get('top_p_bound', 0.0)
    if not 0.0 <= top_p_bound <= 1.0:
        raise ValueError('top_p_bound must be less than or equal to 1.0')
    add_BOS = kwargs.get('add_BOS', False)
    if not isinstance(add_BOS, bool):
        raise ValueError('add_BOS must be a boolean')
    beam_width = num_beams
    if beam_width is not None:
        if not isinstance(beam_width, int):
            raise ValueError('beam_width must be an integer')
        if beam_width < 1:
            raise ValueError('beam_width must be greater than 0')
        if inputs.shape[0] > 1:
            return 'When doing beam_search, batch size must be 1'
    tokenizer = get_tokenizer()
    stop_token = kwargs.get('stop_token', tokenizer.eod)
    if stop_token is not None:
        if not isinstance(stop_token, int):
            raise ValueError('stop_token must be an integer')
    if length_penalty is None:
        length_penalty = 1.0
    sizes_list = None
    prompts_tokens_tensor = None
    prompts_length_tensor = None
    if torch.distributed.get_rank() == 0:
        if attention_mask is None:
            prompts_length_tensor = torch.cuda.LongTensor([inputs.shape[1]] * inputs.shape[0])
        else:
            prompts_length_tensor = attention_mask.sum(axis=-1).cuda()
        if max_new_tokens is None:
            max_new_tokens = max_length - inputs.shape[1]
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be greater than 0')
        if add_BOS:
            max_length = max_new_tokens + inputs.shape[1] + 1
            max_length = 4 * math.ceil(max_length / 4)
            max_new_tokens = max_length - (inputs.shape[1] + 1)
            padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
            prompts_tokens_tensor = torch.concat([torch.unsqueeze(padding[:, 0], axis=-1), inputs.cuda(), padding], axis=-1)
        else:
            max_length = max_new_tokens + inputs.shape[1]
            max_length = 4 * math.ceil(max_length / 4)
            max_new_tokens = max_length - inputs.shape[1]
            padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
            prompts_tokens_tensor = torch.concat([inputs.cuda(), padding], axis=-1)
        sizes_list = [prompts_tokens_tensor.size(0), prompts_tokens_tensor.size(1)]
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)
    sizes = sizes_tensor.tolist()
    context_tokens_tensor = broadcast_tensor(sizes, torch.int64, tensor=prompts_tokens_tensor, rank=0)
    context_length_tensor = broadcast_tensor(sizes[0], torch.int64, tensor=prompts_length_tensor, rank=0)
    random_seed = kwargs.get('random_seed', 0)
    torch.random.manual_seed(random_seed)
    unwrapped_model = unwrap_model(self.base_model, (torchDDP, LocalDDP, Float16Module))
    if beam_width is not None:
        tokens, _ = beam_search_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, beam_width, stop_token=stop_token, num_return_gen=1, length_penalty=length_penalty)
    else:
        tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, return_output_log_probs=False, top_k=top_k, top_p=top_p, top_p_decay=top_p_decay, top_p_bound=top_p_bound, temperature=temperature, use_eod_token_for_early_termination=True)
    return tokens