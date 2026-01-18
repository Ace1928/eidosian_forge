import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.l2_normalize', 'linalg.l2_normalize', 'nn.l2_normalize', v1=['math.l2_normalize', 'linalg.l2_normalize', 'nn.l2_normalize'])
@dispatch.add_dispatch_support
@deprecated_args(None, 'dim is deprecated, use axis instead', 'dim')
def l2_normalize(x, axis=None, epsilon=1e-12, name=None, dim=None):
    """Normalizes along dimension `axis` using an L2 norm.

  For a 1-D tensor with `axis = 0`, computes

      output = x / sqrt(max(sum(x**2), epsilon))

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `axis`.

  1-D tensor example:
  >>> x = tf.constant([3.0, 4.0])
  >>> tf.math.l2_normalize(x).numpy()
  array([0.6, 0.8], dtype=float32)

  2-D tensor example:
  >>> x = tf.constant([[3.0], [4.0]])
  >>> tf.math.l2_normalize(x, 0).numpy()
  array([[0.6],
       [0.8]], dtype=float32)

  >>> x = tf.constant([[3.0], [4.0]])
  >>> tf.math.l2_normalize(x, 1).numpy()
  array([[1.],
       [1.]], dtype=float32)

  Args:
    x: A `Tensor`.
    axis: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).
    dim: Deprecated, do not use.

  Returns:
    A `Tensor` with the same shape as `x`.
  """
    axis = deprecated_argument_lookup('axis', axis, 'dim', dim)
    with ops.name_scope(name, 'l2_normalize', [x]) as name:
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.is_complex:
            square_real = math_ops.square(math_ops.real(x))
            square_imag = math_ops.square(math_ops.imag(x))
            square_sum = math_ops.real(math_ops.reduce_sum(square_real + square_imag, axis, keepdims=True))
            x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
            norm_real = math_ops.multiply(math_ops.real(x), x_inv_norm)
            norm_imag = math_ops.multiply(math_ops.imag(x), x_inv_norm)
            return math_ops.complex(norm_real, norm_imag, name=name)
        square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
        x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
        return math_ops.multiply(x, x_inv_norm, name=name)