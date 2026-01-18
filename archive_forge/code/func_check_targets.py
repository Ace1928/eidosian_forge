from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
def check_targets(self, targets):
    if targets is None:
        raise ValueError('targets should not be None')
    if not all(['boxes' in t for t in targets]):
        raise ValueError('Every element of targets should have a boxes key')
    if not all(['labels' in t for t in targets]):
        raise ValueError('Every element of targets should have a labels key')
    if self.has_mask():
        if not all(['masks' in t for t in targets]):
            raise ValueError('Every element of targets should have a masks key')