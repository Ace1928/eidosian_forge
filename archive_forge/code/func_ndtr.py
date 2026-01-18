import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def ndtr(x, name='ndtr'):
    """Normal distribution function.

  Returns the area under the Gaussian probability density function, integrated
  from minus infinity to x:

  ```
                    1       / x
     ndtr(x)  = ----------  |    exp(-0.5 t**2) dt
                sqrt(2 pi)  /-inf

              = 0.5 (1 + erf(x / sqrt(2)))
              = 0.5 erfc(x / sqrt(2))
  ```

  Args:
    x: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="ndtr").

  Returns:
    ndtr: `Tensor` with `dtype=x.dtype`.

  Raises:
    TypeError: if `x` is not floating-type.
  """
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.as_numpy_dtype not in [np.float32, np.float64]:
            raise TypeError('x.dtype=%s is not handled, see docstring for supported types.' % x.dtype)
        return _ndtr(x)