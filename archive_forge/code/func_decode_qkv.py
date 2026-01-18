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
def decode_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
    curr_ctx = hidden_states.shape[1]
    query = hidden_states
    if sample:
        if self.sample_t == 0:
            self.cache['key'], self.cache['value'] = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
        key, value = (self.cache['key'], self.cache['value'])
        self.sample_t += curr_ctx
    else:
        key, value = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
    return (query, key, value, sample)