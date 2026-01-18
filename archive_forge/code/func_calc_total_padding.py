import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def calc_total_padding(func, was_same, padding, dilation, kernel_size):
    if was_same:
        all_padding = int_padding_for_string_padding(func, 'same', dilation, kernel_size)
        total_padding = tuple((all_padding[i] + all_padding[i - 1] for i in range(len(all_padding) - 1, -1, -2)))
        return total_padding
    else:
        return tuple((2 * pad for pad in padding))