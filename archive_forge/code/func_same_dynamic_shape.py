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
def same_dynamic_shape(a, b):
    """Returns whether a and b have the same dynamic shape.

  Args:
    a: `Tensor`
    b: `Tensor`

  Returns:
    `bool` `Tensor` representing if both tensors have the same shape.
  """
    a = ops.convert_to_tensor(a, name='a')
    b = ops.convert_to_tensor(b, name='b')

    def all_shapes_equal():
        return math_ops.reduce_all(math_ops.equal(array_ops.concat([array_ops.shape(a), array_ops.shape(b)], 0), array_ops.concat([array_ops.shape(b), array_ops.shape(a)], 0)))
    return tf_cond.cond(math_ops.equal(array_ops.rank(a), array_ops.rank(b)), all_shapes_equal, lambda: constant_op.constant(False))