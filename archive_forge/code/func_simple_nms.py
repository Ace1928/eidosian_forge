from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Applies non-maximum suppression on scores"""
    if nms_radius < 0:
        raise ValueError('Expected positive values for nms_radius')

    def max_pool(x):
        return nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | new_max_mask & ~supp_mask
    return torch.where(max_mask, scores, zeros)