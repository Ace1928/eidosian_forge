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
@tf_export.tf_export('experimental.numpy.average', v1=[])
@np_utils.np_doc('average')
def average(a, axis=None, weights=None, returned=False):
    if axis is not None and (not isinstance(axis, int)):
        raise ValueError(f'Argument `axis` must be an integer. Received axis={axis} (of type {type(axis)})')
    a = np_array_ops.array(a)
    default_float_type = np_utils.result_type(float)
    if weights is None:
        if not np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
            a = a.astype(np_utils.result_type(a.dtype, default_float_type))
        avg = math_ops.reduce_mean(a, axis=axis)
        if returned:
            if axis is None:
                weights_sum = array_ops.size(a)
            else:
                weights_sum = array_ops.shape(a)[axis]
            weights_sum = math_ops.cast(weights_sum, a.dtype)
    else:
        if np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
            out_dtype = np_utils.result_type(a.dtype, weights)
        else:
            out_dtype = np_utils.result_type(a.dtype, weights, default_float_type)
        a = np_array_ops.array(a, out_dtype)
        weights = np_array_ops.array(weights, out_dtype)

        def rank_equal_case():
            control_flow_assert.Assert(math_ops.reduce_all(array_ops.shape(a) == array_ops.shape(weights)), [array_ops.shape(a), array_ops.shape(weights)])
            weights_sum = math_ops.reduce_sum(weights, axis=axis)
            avg = math_ops.reduce_sum(a * weights, axis=axis) / weights_sum
            return (avg, weights_sum)
        if axis is None:
            avg, weights_sum = rank_equal_case()
        else:

            def rank_not_equal_case():
                control_flow_assert.Assert(array_ops.rank(weights) == 1, [array_ops.rank(weights)])
                weights_sum = math_ops.reduce_sum(weights)
                axes = ops.convert_to_tensor([[axis], [0]])
                avg = math_ops.tensordot(a, weights, axes) / weights_sum
                return (avg, weights_sum)
            avg, weights_sum = np_utils.cond(math_ops.equal(array_ops.rank(a), array_ops.rank(weights)), rank_equal_case, rank_not_equal_case)
    avg = np_array_ops.array(avg)
    if returned:
        weights_sum = np_array_ops.broadcast_to(weights_sum, array_ops.shape(avg))
        return (avg, weights_sum)
    return avg