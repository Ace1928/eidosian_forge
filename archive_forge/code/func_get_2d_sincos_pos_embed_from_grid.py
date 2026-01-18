import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_vit_mae import ViTMAEConfig
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even')
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb