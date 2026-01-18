import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('shape_n')
@dispatch.add_dispatch_support
def shape_n(input, out_type=dtypes.int32, name=None):
    """Returns shape of a list of tensors.

  Given a list of tensors, `tf.shape_n` is much faster than applying `tf.shape`
  to each tensor individually.
  >>> a = tf.ones([1, 2])
  >>> b = tf.ones([2, 3])
  >>> c = tf.ones([3, 4])
  >>> tf.shape_n([a, b, c])
  [<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>]

  Args:
    input: A list of at least 1 `Tensor` object with the same dtype.
    out_type: The specified output type of the operation (`int32` or `int64`).
      Defaults to `tf.int32`(optional).
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` specifying the shape of each input tensor with type of
    `out_type`.
  """
    return gen_array_ops.shape_n(input, out_type=out_type, name=name)