import torch
import copy
from typing import Dict, Any
def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    if type(module1) not in _all_supported_types or type(module2) not in _all_supported_types:
        raise ValueError('module type not supported:', type(module1), ' ', type(module2))
    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)
    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError('Number of output channels of first arg do not match         number input channels of second arg')
    bias = get_module_bias(module1)
    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)
    weight2_range += 1e-09
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)
    bias = bias * inverse_scaling_factors
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)
    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)
    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors
    set_module_weight(module1, weight1)
    set_module_bias(module1, bias)
    set_module_weight(module2, weight2)