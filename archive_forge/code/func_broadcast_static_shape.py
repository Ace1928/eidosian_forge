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
@tf_export('broadcast_static_shape')
@dispatch.add_dispatch_support
def broadcast_static_shape(shape_x, shape_y):
    """Computes the shape of a broadcast given known shapes.

  When `shape_x` and `shape_y` are fully known `TensorShape`s this computes a
  `TensorShape` which is the shape of the result of a broadcasting op applied in
  tensors of shapes `shape_x` and `shape_y`.

  For example, if shape_x is `TensorShape([1, 2, 3])` and shape_y is
  `TensorShape([5, 1, 3])`, the result is a TensorShape whose value is
  `TensorShape([5, 2, 3])`.

  This is useful when validating the result of a broadcasting operation when the
  tensors have statically known shapes.

  Example:

  >>> shape_x = tf.TensorShape([1, 2, 3])
  >>> shape_y = tf.TensorShape([5, 1 ,3])
  >>> tf.broadcast_static_shape(shape_x, shape_y)
  TensorShape([5, 2, 3])

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
    return common_shapes.broadcast_shape(shape_x, shape_y)