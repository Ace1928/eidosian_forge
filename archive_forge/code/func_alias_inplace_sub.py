from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_sub, which offers the same functionality with well-defined read-write semantics.')
def alias_inplace_sub(x, i, v):
    """Applies an inplace sub on input x at index i with value v. Aliases x.

  If i is None, x and v must be the same shape. Computes
    x -= v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] -= v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] -= v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns x.

  """
    return _inplace_helper(x, i, v, gen_array_ops.inplace_sub)