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
def factored_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
    curr_ctx = hidden_states.shape[1]
    if last_encoder_hidden_states is not None:
        raise TypeError('last_encoder_hidden_states should be None')
    query, key, value = hidden_states.chunk(3, dim=2)
    if sample:
        self.sample_t += curr_ctx
        key, value = self._append_cache(key, value)
        l_cache = self._suff_cache_len()
        if self._cache_len() > l_cache:
            self._slice_cache(-l_cache)
        if curr_ctx > 1:
            if self.attn_func != 'dense_attn':
                query = self._pad_to_block_ctx(query, query=True)
                key = self._pad_to_block_ctx(key)
                value = self._pad_to_block_ctx(value)
            sample = False
        else:
            key = self.cache['key']
            value = self.cache['value']
    return (query, key, value, sample)