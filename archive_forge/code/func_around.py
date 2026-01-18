import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.around', v1=[])
@np_utils.np_doc('around')
def around(a, decimals=0):
    a = asarray(a)
    dtype = a.dtype.as_numpy_dtype
    factor = math.pow(10, decimals)
    if np.issubdtype(dtype, np.inexact):
        factor = math_ops.cast(factor, dtype)
    else:
        float_dtype = np_utils.result_type(float)
        a = a.astype(float_dtype)
        factor = math_ops.cast(factor, float_dtype)
    a = math_ops.multiply(a, factor)
    a = math_ops.round(a)
    a = math_ops.divide(a, factor)
    return a.astype(dtype)