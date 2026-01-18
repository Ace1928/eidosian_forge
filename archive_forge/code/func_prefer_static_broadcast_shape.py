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
def prefer_static_broadcast_shape(shape1, shape2, name='prefer_static_broadcast_shape'):
    """Convenience function which statically broadcasts shape when possible.

  Args:
    shape1:  `1-D` integer `Tensor`.  Already converted to tensor!
    shape2:  `1-D` integer `Tensor`.  Already converted to tensor!
    name:  A string name to prepend to created ops.

  Returns:
    The broadcast shape, either as `TensorShape` (if broadcast can be done
      statically), or as a `Tensor`.
  """
    with ops.name_scope(name, values=[shape1, shape2]):

        def make_shape_tensor(x):
            return ops.convert_to_tensor(x, name='shape', dtype=dtypes.int32)

        def get_tensor_shape(s):
            if isinstance(s, tensor_shape.TensorShape):
                return s
            s_ = tensor_util.constant_value(make_shape_tensor(s))
            if s_ is not None:
                return tensor_shape.TensorShape(s_)
            return None

        def get_shape_tensor(s):
            if not isinstance(s, tensor_shape.TensorShape):
                return make_shape_tensor(s)
            if s.is_fully_defined():
                return make_shape_tensor(s.as_list())
            raise ValueError('Cannot broadcast from partially defined `TensorShape`.')
        shape1_ = get_tensor_shape(shape1)
        shape2_ = get_tensor_shape(shape2)
        if shape1_ is not None and shape2_ is not None:
            return array_ops.broadcast_static_shape(shape1_, shape2_)
        shape1_ = get_shape_tensor(shape1)
        shape2_ = get_shape_tensor(shape2)
        return array_ops.broadcast_dynamic_shape(shape1_, shape2_)