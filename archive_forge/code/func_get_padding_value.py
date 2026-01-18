import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
def get_padding_value(padding=None, kernel_size=7, stride=1, dilation=1) -> Tuple[Tuple, bool]:
    """
    Utility function to get the tuple padding value given the kernel_size and padding.

    Args:
        padding (Union[`str`, `int`], *optional*):
            Padding value, can be either `"same"`, `"valid"`. If a different value is provided the default padding from
            PyTorch is used.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size of the convolution layers.
        stride (`int`, *optional*, defaults to 1):
            Stride value of the convolution layers.
        dilation (`int`, *optional*, defaults to 1):
            Dilation value of the convolution layers.
    """
    dynamic = False
    if padding is None:
        padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
        return (padding, dynamic)
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if stride == 1 and dilation * (kernel_size - 1) % 2 == 0:
                padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return (padding, dynamic)