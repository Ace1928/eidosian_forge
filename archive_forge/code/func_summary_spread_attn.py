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
def summary_spread_attn(self, query, key, value, sample):
    blocks = self.blocks
    spread = self.spread
    batch_size, seq_len, embed_dim = value.shape
    if sample:
        raise NotImplementedError
    else:
        key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
        key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0)).contiguous()
        key = key.view(batch_size, blocks * spread, embed_dim)
        value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
        value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0)).contiguous()
        value = value.view(batch_size, blocks * spread, embed_dim)
        return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)