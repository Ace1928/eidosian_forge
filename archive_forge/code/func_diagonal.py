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
@tf_export.tf_export('experimental.numpy.diagonal', v1=[])
@np_utils.np_doc('diagonal')
def diagonal(a, offset=0, axis1=0, axis2=1):
    a = asarray(a)
    maybe_rank = a.shape.rank
    if maybe_rank is not None and offset == 0 and (axis1 == maybe_rank - 2 or axis1 == -2) and (axis2 == maybe_rank - 1 or axis2 == -1):
        return array_ops.matrix_diag_part(a)
    a = moveaxis(a, (axis1, axis2), (-2, -1))
    a_shape = array_ops.shape(a)

    def _zeros():
        return (array_ops.zeros(array_ops.concat([a_shape[:-1], [0]], 0), dtype=a.dtype), 0)
    a, offset = np_utils.cond(np_utils.logical_or(np_utils.less_equal(offset, -1 * np_utils.getitem(a_shape, -2)), np_utils.greater_equal(offset, np_utils.getitem(a_shape, -1))), _zeros, lambda: (a, offset))
    a = array_ops.matrix_diag_part(a, k=offset)
    return a