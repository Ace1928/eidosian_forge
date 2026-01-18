from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def pool_features(self, x: Tensor, attn_mask: Tensor) -> Tensor:
    if self.pooling == 'cls':
        return x[:, 0]
    attn_mask = attn_mask.unsqueeze(2).type_as(x)
    return (x * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)