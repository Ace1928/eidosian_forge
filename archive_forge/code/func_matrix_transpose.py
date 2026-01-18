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
@tf_export('linalg.matrix_transpose', v1=['linalg.transpose', 'linalg.matrix_transpose', 'matrix_transpose'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('matrix_transpose', 'linalg.transpose')
def matrix_transpose(a, name='matrix_transpose', conjugate=False):
    """Transposes last two dimensions of tensor `a`.

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.linalg.matrix_transpose(x)  # [[1, 4],
                                 #  [2, 5],
                                 #  [3, 6]]

  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.linalg.matrix_transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                                 #  [2 - 2j, 5 - 5j],
                                                 #  [3 - 3j, 6 - 6j]]

  # Matrix with two batch dimensions.
  # x.shape is [1, 2, 3, 4]
  # tf.linalg.matrix_transpose(x) is shape [1, 2, 4, 3]
  ```

  Note that `tf.matmul` provides kwargs allowing for transpose of arguments.
  This is done with minimal cost, and is preferable to using this function. E.g.

  ```python
  # Good!  Transpose is taken at minimal additional cost.
  tf.matmul(matrix, b, transpose_b=True)

  # Inefficient!
  tf.matmul(matrix, tf.linalg.matrix_transpose(b))
  ```

  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.

  TensorFlow does not support strides, `linalg.matrix_transpose` returns a new
  tensor with the items permuted.
  @end_compatibility

  Args:
    a: A `Tensor` with `rank >= 2`.
    name: A name for the operation (optional).
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.linalg.matrix_transpose(input)).

  Returns:
    A transposed batch matrix `Tensor`.

  Raises:
    ValueError:  If `a` is determined statically to have `rank < 2`.
  """
    with ops.name_scope(name, values=[a]):
        a = ops.convert_to_tensor(a, name='a')
        a_shape = a.get_shape()
        ndims = a_shape.ndims
        if ndims is not None:
            if ndims < 2:
                raise ValueError(f'Argument `a` should be a (batch) matrix with rank >= 2.  Received `a` = {a} with shape: {a_shape}')
            perm = list(range(ndims - 2)) + [ndims - 1] + [ndims - 2]
        else:
            a_rank = rank(a)
            perm = concat((gen_math_ops._range(0, a_rank - 2, 1), [a_rank - 1, a_rank - 2]), 0)
        return transpose(a, perm=perm, conjugate=conjugate)