import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def conv_connected_inputs(input_shape, kernel_shape, output_position, strides, padding):
    """Return locations of the input connected to an output position.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)`. This method
  returns N ranges specifying the input region that was convolved with the
  kernel to produce the output at position
  `output_position = (p_out1, ..., p_outN)`.

  Example:

    >>> input_shape = (4, 4)
    >>> kernel_shape = (2, 1)
    >>> output_position = (1, 1)
    >>> strides = (1, 1)
    >>> padding = "valid"
    >>> conv_connected_inputs(input_shape, kernel_shape, output_position,
    ...                       strides, padding)
    [range(1, 3), range(1, 2)]

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    output_position: tuple of size N: `(p_out1, ..., p_outN)`, a single position
      in the output of the convolution.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    N ranges `[[p_in_left1, ..., p_in_right1], ...,
              [p_in_leftN, ..., p_in_rightN]]` specifying the region in the
    input connected to output_position.
  """
    ranges = []
    ndims = len(input_shape)
    for d in range(ndims):
        left_shift = int(kernel_shape[d] / 2)
        right_shift = kernel_shape[d] - left_shift
        center = output_position[d] * strides[d]
        if padding == 'valid':
            center += left_shift
        start = max(0, center - left_shift)
        end = min(input_shape[d], center + right_shift)
        ranges.append(range(start, end))
    return ranges