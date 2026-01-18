import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.argmax', 'argmax', v1=[])
@dispatch.add_dispatch_support
def argmax_v2(input, axis=None, output_type=dtypes.int64, name=None):
    """Returns the index with the largest value across axes of a tensor.

  In case of identity returns the smallest index.

  For example:

  >>> A = tf.constant([2, 20, 30, 3, 6])
  >>> tf.math.argmax(A)  # A[2] is maximum in tensor A
  <tf.Tensor: shape=(), dtype=int64, numpy=2>
  >>> B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
  ...                  [14, 45, 23, 5, 27]])
  >>> tf.math.argmax(B, 0)
  <tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2])>
  >>> tf.math.argmax(B, 1)
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 1])>
  >>> C = tf.constant([0, 0, 0, 0])
  >>> tf.math.argmax(C) # Returns smallest index in case of ties
  <tf.Tensor: shape=(), dtype=int64, numpy=0>

  Args:
    input: A `Tensor`.
    axis: An integer, the axis to reduce across. Default to 0.
    output_type: An optional output dtype (`tf.int32` or `tf.int64`). Defaults
      to `tf.int64`.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of type `output_type`.
  """
    if axis is None:
        axis = 0
    return gen_math_ops.arg_max(input, axis, name=name, output_type=output_type)