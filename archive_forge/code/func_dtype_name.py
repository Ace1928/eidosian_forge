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
def dtype_name(dtype):
    """Returns the string name for this `dtype`."""
    dtype = dtypes.as_dtype(dtype)
    if hasattr(dtype, 'name'):
        return dtype.name
    if hasattr(dtype, '__name__'):
        return dtype.__name__
    return str(dtype)