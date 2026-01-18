from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList
def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer