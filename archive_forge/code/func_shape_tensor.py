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
def shape_tensor(shape, name=None):
    """Convert Tensor using default type, unless empty list or tuple."""
    if isinstance(shape, (tuple, list)) and (not shape):
        dtype = dtypes.int32
    else:
        dtype = None
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(shape, dtype=dtype, name=name)