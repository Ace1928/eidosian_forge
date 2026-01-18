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
@staticmethod
def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / half_dim
    emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
    emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    return (torch.sin(emb), torch.cos(emb))