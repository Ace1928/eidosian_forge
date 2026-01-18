import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor

    Args:
        input (`torch.Tensor`):
            The target tensor to calculate percentile max and min.
        lower_percentile (`float`):
            If 0.1, means we return the value of the smallest 0.1% value in the tensor as percentile min.
        upper_percentile (`float`):
            If 99.9, means we return the value of the largest 0.1% value in the tensor as percentile max.
        output_tensor (`bool`, *optional*, defaults to `False`):
            If True, this function returns tensors, otherwise it returns values.

    Returns:
        `Tuple(torch.Tensor, torch.Tensor)`: Percentile min and max value of *input*
    """
    input_length = input.shape[0]
    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)
    upper_bound = torch.kthvalue(input, k=upper_index).values
    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values
    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return (lower_bound, upper_bound)