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
@tf_export('nn.avg_pool', v1=['nn.avg_pool_v2'])
@dispatch.add_dispatch_support
def avg_pool_v2(input, ksize, strides, padding, data_format=None, name=None):
    """Performs the avg pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    input:  Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if `data_format` does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC". Pooling happens over the spatial dimensions only.
    ksize: An int or list of `ints` that has length `1`, `N` or `N+2`. The size
      of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string. Specifies the channel dimension. For N=1 it can be
      either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default)
      or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
    name: Optional name for the operation.

  Returns:
    A `Tensor` of format specified by `data_format`.
    The average pooled output tensor.
  """
    if input.shape is not None:
        n = len(input.shape) - 2
    elif data_format is not None:
        n = len(data_format) - 2
    else:
        raise ValueError(f'`input` must have a static shape or `data_format` must be given. Received: input.shape={input.shape} and data_format={data_format}')
    if not 1 <= n <= 3:
        raise ValueError(f'`input.shape.rank` must be 3, 4 or 5. Received: input.shape={input.shape} of rank {n + 2}.')
    if data_format is None:
        channel_index = n + 1
    else:
        channel_index = 1 if data_format.startswith('NC') else n + 1
    ksize = _get_sequence(ksize, n, channel_index, 'ksize')
    strides = _get_sequence(strides, n, channel_index, 'strides')
    avg_pooling_ops = {1: avg_pool1d, 2: gen_nn_ops.avg_pool, 3: gen_nn_ops.avg_pool3d}
    op = avg_pooling_ops[n]
    return op(input, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)