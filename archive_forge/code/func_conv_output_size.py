import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def conv_output_size(input_size: List[int], weight_size: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
    check_shape_forward(input_size, weight_size, bias, stride, padding, dilation, groups)
    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size: List[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size.append((input_size[d] + 2 * padding[d - 2] - kernel) // stride[d - 2] + 1)
    return output_size