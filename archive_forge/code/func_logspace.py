import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.logspace', v1=[])
@np_utils.np_doc('logspace')
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    dtype = np_utils.result_type(start, stop, dtype)
    result = linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)
    result = math_ops.pow(math_ops.cast(base, result.dtype), result)
    if dtype:
        result = math_ops.cast(result, dtype)
    return result