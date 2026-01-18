import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.is_numeric_tensor', v1=['debugging.is_numeric_tensor', 'is_numeric_tensor'])
@deprecation.deprecated_endpoints('is_numeric_tensor')
def is_numeric_tensor(tensor):
    """Returns `True` if the elements of `tensor` are numbers.

  Specifically, returns `True` if the dtype of `tensor` is one of the following:

  * `tf.float16`
  * `tf.float32`
  * `tf.float64`
  * `tf.int8`
  * `tf.int16`
  * `tf.int32`
  * `tf.int64`
  * `tf.uint8`
  * `tf.uint16`
  * `tf.uint32`
  * `tf.uint64`
  * `tf.qint8`
  * `tf.qint16`
  * `tf.qint32`
  * `tf.quint8`
  * `tf.quint16`
  * `tf.complex64`
  * `tf.complex128`
  * `tf.bfloat16`

  Returns `False` if `tensor` is of a non-numeric type or if `tensor` is not
  a `tf.Tensor` object.
  """
    return isinstance(tensor, tensor_lib.Tensor) and tensor.dtype in NUMERIC_TYPES