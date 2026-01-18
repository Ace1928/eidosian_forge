from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_add, which offers the same functionality with well-defined read-write semantics.')
def inplace_add(x, i, v):
    """Applies an inplace add on input x at index i with value v.

  Note that this function is not actually inplace - it allocates
  a copy of x.  The utility is not avoiding memory copies but rather
  specifying a sparse update.

  If i is None, x and v must be the same shape. Computes
    y = x; y += v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    y = x; y[i, :] += v;
  Otherwise, x and v must have the same rank. Computes
    y = x; y[i, :] += v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns y, which is guaranteed not to be an alias of x.

  """
    return alias_inplace_add(gen_array_ops.deep_copy(x), i, v)