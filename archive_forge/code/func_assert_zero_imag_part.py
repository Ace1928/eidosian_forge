import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def assert_zero_imag_part(x, message=None, name='assert_zero_imag_part'):
    """Returns `Op` that asserts Tensor `x` has no non-zero imaginary parts.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
    with ops.name_scope(name, values=[x]):
        x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name='x')
        dtype = x.dtype.base_dtype
        if dtype.is_floating:
            return control_flow_ops.no_op()
        zero = tensor_conversion.convert_to_tensor_v2_with_dispatch(0, dtype=dtype.real_dtype)
        return check_ops.assert_equal(zero, math_ops.imag(x), message=message)