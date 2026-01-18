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
def prime_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
    curr_ctx = hidden_states.shape[1]
    if last_encoder_hidden_states is not None:
        raise TypeError('last_encoder_hidden_states should be None')
    query, key, value = hidden_states.chunk(3, dim=2)
    if sample:
        if self._cache_len() < self._encoder_len:
            self._append_cache(key, value)
        if self._cache_len() > self._encoder_len:
            self._slice_cache(0, self._encoder_len)
        key, value = (self.cache['key'], self.cache['value'])
        self.sample_t += curr_ctx
    return (query, key, value, sample)