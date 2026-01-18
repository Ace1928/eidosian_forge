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