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
def prefer_static_rank(x):
    """Return static rank of tensor `x` if available, else `tf.rank(x)`.

  Args:
    x: `Tensor` (already converted).

  Returns:
    Numpy array (if static rank is obtainable), else `Tensor`.
  """
    return prefer_static_value(array_ops.rank(x))