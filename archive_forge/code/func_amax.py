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
@tf_export.tf_export('experimental.numpy.amax', v1=[])
@np_utils.np_doc('amax', unsupported_params=['out'])
def amax(a, axis=None, out=None, keepdims=None):
    if out is not None:
        raise ValueError('Setting out is not supported.')
    return _reduce(math_ops.reduce_max, a, axis=axis, dtype=None, keepdims=keepdims, promote_int=None, tf_bool_fn=math_ops.reduce_any, preserve_bool=True)