import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
def same_padding(in_size: Tuple[int, int], filter_size: Union[int, Tuple[int, int]], stride_size: Union[int, Tuple[int, int]]) -> (Union[int, Tuple[int, int]], Tuple[int, int]):
    """Note: Padding is added to match TF conv2d `same` padding.

    See www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size: Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size: Rows (Height), column (Width) for filter

    Returns:
        padding: For input into torch.nn.ZeroPad2d.
        output: Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = (filter_size, filter_size)
    else:
        filter_height, filter_width = filter_size
    if isinstance(stride_size, (int, float)):
        stride_height, stride_width = (int(stride_size), int(stride_size))
    else:
        stride_height, stride_width = (int(stride_size[0]), int(stride_size[1]))
    out_height = int(np.ceil(float(in_height) / float(stride_height)))
    out_width = int(np.ceil(float(in_width) / float(stride_width)))
    pad_along_height = int((out_height - 1) * stride_height + filter_height - in_height)
    pad_along_width = int((out_width - 1) * stride_width + filter_width - in_width)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return (padding, output)