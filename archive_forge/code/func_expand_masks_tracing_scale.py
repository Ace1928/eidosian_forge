from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)