import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_input_for_string_padding(func, padding_style, input, dilation, kernel_size):
    if padding_style == 'valid':
        return input
    else:
        padding = int_padding_for_string_padding(func, padding_style, dilation, kernel_size)
        return F.pad(input, padding)