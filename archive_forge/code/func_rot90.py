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
@tf_export.tf_export('experimental.numpy.rot90', v1=[])
@np_utils.np_doc('rot90')
def rot90(m, k=1, axes=(0, 1)):
    m_rank = array_ops.rank(m)
    ax1, ax2 = np_utils._canonicalize_axes(axes, m_rank)
    k = k % 4
    if k == 0:
        return m
    elif k == 2:
        return flip(flip(m, ax1), ax2)
    else:
        perm = math_ops.range(m_rank)
        perm = array_ops.tensor_scatter_update(perm, [[ax1], [ax2]], [ax2, ax1])
        if k == 1:
            return transpose(flip(m, ax2), perm)
        else:
            return flip(transpose(m, perm), ax2)