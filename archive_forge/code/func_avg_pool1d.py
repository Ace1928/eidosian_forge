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
@tf_export('nn.avg_pool1d')
@dispatch.add_dispatch_support
def avg_pool1d(input, ksize, strides, padding, data_format='NWC', name=None):
    """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Note internally this op reshapes and uses the underlying 2d operation.

  Args:
    input: A 3-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1` or `3`. The size of the
      window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1` or `3`. The stride of
      the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
    with ops.name_scope(name, 'AvgPool1D', [input]) as name:
        if data_format is None:
            data_format = 'NWC'
        channel_index = 1 if data_format.startswith('NC') else 2
        ksize = [1] + _get_sequence(ksize, 1, channel_index, 'ksize')
        strides = [1] + _get_sequence(strides, 1, channel_index, 'strides')
        expanding_dim = 1 if data_format == 'NWC' else 2
        data_format = 'NHWC' if data_format == 'NWC' else 'NCHW'
        input = array_ops.expand_dims_v2(input, expanding_dim)
        result = gen_nn_ops.avg_pool(input, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)
        return array_ops.squeeze(result, expanding_dim)