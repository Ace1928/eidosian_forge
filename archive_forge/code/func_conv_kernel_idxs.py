import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def conv_kernel_idxs(input_shape, kernel_shape, strides, padding, filters_in, filters_out, data_format):
    """Yields output-input tuples of indices in a CNN layer.

  The generator iterates over all `(output_idx, input_idx)` tuples, where
    `output_idx` is an integer index in a flattened tensor representing a single
    output image of a convolutional layer that is connected (via the layer
    weights) to the respective single input image at `input_idx`

  Example:

    >>> input_shape = (2, 2)
    >>> kernel_shape = (2, 1)
    >>> strides = (1, 1)
    >>> padding = "valid"
    >>> filters_in = 1
    >>> filters_out = 1
    >>> data_format = "channels_last"
    >>> list(conv_kernel_idxs(input_shape, kernel_shape, strides, padding,
    ...                       filters_in, filters_out, data_format))
    [(0, 0), (0, 2), (1, 1), (1, 3)]

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
    filters_in: `int`, number if filters in the input to the layer.
    filters_out: `int', number if filters in the output of the layer.
    data_format: string, "channels_first" or "channels_last".

  Yields:
    The next tuple `(output_idx, input_idx)`, where
    `output_idx` is an integer index in a flattened tensor representing a single
    output image of a convolutional layer that is connected (via the layer
    weights) to the respective single input image at `input_idx`.

  Raises:
      ValueError: if `data_format` is neither
      `"channels_last"` nor `"channels_first"`, or if number of strides, input,
      and kernel number of dimensions do not match.

      NotImplementedError: if `padding` is neither `"same"` nor `"valid"`.
  """
    if padding not in ('same', 'valid'):
        raise NotImplementedError('Padding type %s not supported. Only "valid" and "same" are implemented.' % padding)
    in_dims = len(input_shape)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * in_dims
    if isinstance(strides, int):
        strides = (strides,) * in_dims
    kernel_dims = len(kernel_shape)
    stride_dims = len(strides)
    if kernel_dims != in_dims or stride_dims != in_dims:
        raise ValueError('Number of strides, input and kernel dimensions must all match. Received: %d, %d, %d.' % (stride_dims, in_dims, kernel_dims))
    output_shape = conv_output_shape(input_shape, kernel_shape, strides, padding)
    output_axes_ticks = [range(dim) for dim in output_shape]
    if data_format == 'channels_first':
        concat_idxs = lambda spatial_idx, filter_idx: (filter_idx,) + spatial_idx
    elif data_format == 'channels_last':
        concat_idxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)
    else:
        raise ValueError('Data format %s not recognized.`data_format` must be "channels_first" or "channels_last".' % data_format)
    for output_position in itertools.product(*output_axes_ticks):
        input_axes_ticks = conv_connected_inputs(input_shape, kernel_shape, output_position, strides, padding)
        for input_position in itertools.product(*input_axes_ticks):
            for f_in in range(filters_in):
                for f_out in range(filters_out):
                    out_idx = np.ravel_multi_index(multi_index=concat_idxs(output_position, f_out), dims=concat_idxs(output_shape, filters_out))
                    in_idx = np.ravel_multi_index(multi_index=concat_idxs(input_position, f_in), dims=concat_idxs(input_shape, filters_in))
                    yield (out_idx, in_idx)