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
@tf_export.tf_export('experimental.numpy.diag_indices', v1=[])
@np_utils.np_doc('diag_indices')
def diag_indices(n, ndim=2):
    if n < 0:
        raise ValueError('n argument to diag_indices must be nonnegative, got {}'.format(n))
    if ndim < 0:
        raise ValueError('ndim argument to diag_indices must be nonnegative, got {}'.format(ndim))
    return (math_ops.range(n),) * ndim