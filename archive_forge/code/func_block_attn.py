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
def block_attn(self, query, key, value, sample):
    block_ctx = self.block_ctx
    batch_size, seq_len, embed_dim = value.shape
    if sample:
        return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
    else:
        query_length = query.shape[1]
        query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
        if query_length < seq_len:
            seq_len = query_length
            key = key[:, -seq_len:].contiguous()
            value = value[:, -seq_len:].contiguous()
        key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
        value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
        return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)