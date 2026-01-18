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
def prime_attn(self, query, key, value, sample):
    encoder_len = self._encoder_len
    key = key[:, :encoder_len]
    value = value[:, :encoder_len]
    return self.dense_attn(query, key, value, sample)