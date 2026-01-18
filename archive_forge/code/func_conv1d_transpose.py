import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.conv1d_transpose')
@dispatch.add_dispatch_support
def conv1d_transpose(input, filters, output_shape, strides, padding='SAME', data_format='NWC', dilations=None, name=None):
    """The transpose of `conv1d`.

  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is actually the transpose (gradient) of `conv1d`
  rather than an actual deconvolution.

  Args:
    input: A 3-D `Tensor` of type `float` and shape
      `[batch, in_width, in_channels]` for `NWC` data format or
      `[batch, in_channels, in_width]` for `NCW` data format.
    filters: A 3-D `Tensor` with the same type as `input` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `input`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `input`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, if
      `output_shape` is not at 3-element vector, if `padding` is other than
      `'VALID'` or `'SAME'`, or if `data_format` is invalid.

  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
    with ops.name_scope(name, 'conv1d_transpose', [input, filters, output_shape]) as name:
        if data_format is None or data_format == 'NWC':
            data_format = 'NHWC'
            spatial_start_dim = 1
            channel_index = 2
        elif data_format == 'NCW':
            data_format = 'NCHW'
            spatial_start_dim = 2
            channel_index = 1
        else:
            raise ValueError(f"`data_format` must be 'NWC' or 'NCW'. Received: data_format={data_format}")
        strides = [1] + _get_sequence(strides, 1, channel_index, 'stride')
        dilations = [1] + _get_sequence(dilations, 1, channel_index, 'dilations')
        input = array_ops.expand_dims(input, spatial_start_dim)
        filters = array_ops.expand_dims(filters, 0)
        output_shape = list(output_shape) if not isinstance(output_shape, tensor_lib.Tensor) else output_shape
        output_shape = array_ops.concat([output_shape[:spatial_start_dim], [1], output_shape[spatial_start_dim:]], 0)
        result = gen_nn_ops.conv2d_backprop_input(input_sizes=output_shape, filter=filters, out_backprop=input, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)
        return array_ops.squeeze(result, spatial_start_dim)