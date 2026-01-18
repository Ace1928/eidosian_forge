from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
def has_keypoint(self):
    if self.keypoint_roi_pool is None:
        return False
    if self.keypoint_head is None:
        return False
    if self.keypoint_predictor is None:
        return False
    return True