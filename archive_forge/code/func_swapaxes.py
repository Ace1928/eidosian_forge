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
@tf_export.tf_export('experimental.numpy.swapaxes', v1=[])
@np_utils.np_doc('swapaxes')
def swapaxes(a, axis1, axis2):
    a = asarray(a)

    def adjust_axes(axes, rank):

        def f(x):
            if isinstance(x, int):
                if x < 0:
                    x = x + rank
            else:
                x = array_ops.where_v2(x < 0, np_utils.add(x, a_rank), x)
            return x
        return nest.map_structure(f, axes)
    if a.shape.rank is not None and isinstance(axis1, int) and isinstance(axis2, int):
        a_rank = a.shape.rank
        axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
        perm = list(range(a_rank))
        perm[axis1] = axis2
        perm[axis2] = axis1
    else:
        a_rank = array_ops.rank(a)
        axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
        perm = math_ops.range(a_rank)
        perm = array_ops.tensor_scatter_update(perm, [[axis1], [axis2]], [axis2, axis1])
    a = array_ops.transpose(a, perm)
    return a