import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def rotary(self, input):
    seq_len, embed_dim = input.size()
    chunk_1, chunk_2 = torch.chunk(input, 2, dim=-1)
    if self.sine is None or seq_len > self.sine.size(0):
        self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(seq_len, embed_dim)
        self.max_positions = seq_len
    self.sine = self.sine.to(self._float_tensor)
    self.cosine = self.cosine.to(self._float_tensor)
    sin = self.sine[:seq_len]
    cos = self.cosine[:seq_len]
    return torch.cat([chunk_1 * cos - chunk_2 * sin, chunk_2 * cos + chunk_1 * sin], dim=1)