import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def maybe_get_static_value(x, dtype=None):
    """Helper which tries to return a static value.

  Given `x`, extract it's value statically, optionally casting to a specific
  dtype. If this is not possible, None is returned.

  Args:
    x: `Tensor` for which to extract a value statically.
    dtype: Optional dtype to cast to.

  Returns:
    Statically inferred value if possible, otherwise None.
  """
    if x is None:
        return x
    try:
        x_ = tensor_util.constant_value(x)
    except TypeError:
        x_ = x
    if x_ is None or dtype is None:
        return x_
    return np.array(x_, dtype)