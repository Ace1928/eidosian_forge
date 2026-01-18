import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.multiply_no_nan')
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply_no_nan(x, y, name=None):
    """Computes the product of x and y and returns 0 if the y is zero, even if x is NaN or infinite.

  Note this is noncommutative: if y is NaN or infinite and x is 0, the result
  will be NaN.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` whose dtype is compatible with `x`.
    name: A name for the operation (optional).

  Returns:
    The element-wise value of the x times y.
  """
    with ops.name_scope(name, 'multiply_no_nan', [x, y]) as name:
        x = ops.convert_to_tensor(x, name='x')
        y = ops.convert_to_tensor(y, name='y', dtype=x.dtype.base_dtype)
        x_dtype = x.dtype.base_dtype
        y_dtype = y.dtype.base_dtype
        if x_dtype != y_dtype:
            raise TypeError(f'`x` and `y` must have the same dtype, got {x_dtype!r} != {y_dtype!r}')
        return gen_math_ops.mul_no_nan(x, y, name=name)