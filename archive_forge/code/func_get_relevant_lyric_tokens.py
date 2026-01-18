import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def get_relevant_lyric_tokens(full_tokens, max_n_lyric_tokens, total_length, offset, duration):
    """
    Extract only the relevant tokens based on the character position. A total of `max_n_lyric_tokens` tokens will be
    returned. If the provided token sequence is smaller, it will be padded, otherwise, only characters ranging from the
    midpoint - `max_n_lyric_tokens//2` to the midpoint + `max_n_lyric_tokens//2` will be returned. This *focuses* on
    the most relevant tokens (in time) for the sequence.

    Args:
        full_tokens (`List[int]`):
            List containing the token ids of the entire lyrics.
        total_length (`int`):
            Total expected length of the music (not all of it is generated, see duration), in samples.
        offset (`int`):
            Starting sample in the music. If the offset is greater than 0, the lyrics will be shifted take that into
            account
        duration (`int`):
            Expected duration of the generated music, in samples. The duration has to be smaller than the total length,
            which represent the overall length of the signal,
    """
    full_tokens = full_tokens[0]
    if len(full_tokens) < max_n_lyric_tokens:
        tokens = torch.cat([torch.zeros(max_n_lyric_tokens - len(full_tokens), dtype=torch.long).to(full_tokens.device), full_tokens])
        indices = [-1] * (max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, max_n_lyric_tokens // 2), len(full_tokens) - max_n_lyric_tokens // 2)
        tokens = full_tokens[midpoint - max_n_lyric_tokens // 2:midpoint + max_n_lyric_tokens // 2]
        indices = list(range(midpoint - max_n_lyric_tokens // 2, midpoint + max_n_lyric_tokens // 2))
    return (tokens.unsqueeze(dim=0), indices)