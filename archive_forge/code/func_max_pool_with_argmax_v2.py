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
@tf_export('nn.max_pool_with_argmax', v1=[])
@dispatch.add_dispatch_support
def max_pool_with_argmax_v2(input, ksize, strides, padding, data_format='NHWC', output_dtype=dtypes.int64, include_batch_in_index=False, name=None):
    """Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
  `include_batch_in_index` is False;
  `((b * height + y) * width + x) * channels + c`
  if `include_batch_in_index` is True.

  The indices returned are always in `[0, height) x [0, width)` before
  flattening, even if padding is involved and the mathematically correct answer
  is outside (either negative or too large).  This is a bug, but fixing it is
  difficult to do in a safe backwards compatible way, especially due to
  flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`,
      `uint32`, `uint64`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`.
      The size of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: An optional `string`, must be set to `"NHWC"`. Defaults to
      `"NHWC"`.
      Specify the data format of the input and output data.
    output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`.
      Defaults to `tf.int64`.
      The dtype of the returned argmax tensor.
    include_batch_in_index: An optional `boolean`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`.
    argmax: A `Tensor` of type `output_dtype`.
  """
    if data_format != 'NHWC':
        raise ValueError(f"`data_format` values other  than 'NHWC' are not supported. Received: data_format={data_format}")
    ksize = _get_sequence(ksize, 2, 3, 'ksize')
    strides = _get_sequence(strides, 2, 3, 'strides')
    return gen_nn_ops.max_pool_with_argmax(input=input, ksize=ksize, strides=strides, padding=padding, Targmax=output_dtype, include_batch_in_index=include_batch_in_index, name=name)