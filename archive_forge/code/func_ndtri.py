import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def ndtri(p, name='ndtri'):
    """The inverse of the CDF of the Normal distribution function.

  Returns x such that the area under the pdf from minus infinity to x is equal
  to p.

  A piece-wise rational approximation is done for the function.
  This is a port of the implementation in netlib.

  Args:
    p: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="ndtri").

  Returns:
    x: `Tensor` with `dtype=p.dtype`.

  Raises:
    TypeError: if `p` is not floating-type.
  """
    with ops.name_scope(name, values=[p]):
        p = ops.convert_to_tensor(p, name='p')
        if p.dtype.as_numpy_dtype not in [np.float32, np.float64]:
            raise TypeError('p.dtype=%s is not handled, see docstring for supported types.' % p.dtype)
        return _ndtri(p)