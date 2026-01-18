import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def conv_output_shape(input_shape, kernel_shape, strides, padding):
    """Return the output shape of an N-D convolution.

  Forces dimensions where input is empty (size 0) to remain empty.

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    tuple of size N: `(d_out1, ..., d_outN)`, spatial shape of the output.
  """
    dims = range(len(kernel_shape))
    output_shape = [conv_output_length(input_shape[d], kernel_shape[d], padding, strides[d]) for d in dims]
    output_shape = tuple([0 if input_shape[d] == 0 else output_shape[d] for d in dims])
    return output_shape