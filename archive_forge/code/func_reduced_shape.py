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
def reduced_shape(input_shape, axes):
    """Helper function for reduction ops.

  Args:
    input_shape: 1-D Tensor, the shape of the Tensor being reduced.
    axes: 1-D Tensor, the reduction axes.

  Returns:
    A 1-D Tensor, the output shape as if keepdims were set to True.
  """
    constant_input_shape = tensor_util.constant_value(input_shape)
    if constant_input_shape is not None:
        constant_axes = tensor_util.constant_value(axes)
        if constant_axes is not None:
            constant_axes = np.array(constant_axes, dtype=np.int32)
            constant_input_shape = np.array(constant_input_shape, dtype=np.int32)
            constant_input_shape[constant_axes] = 1
            return constant_input_shape
    input_shape = cast(input_shape, dtypes.int32)
    axes = cast(axes, dtypes.int32)
    input_rank = array_ops.size(input_shape)
    axes = (axes + input_rank) % input_rank
    axes_shape = array_ops.shape(axes)
    return gen_data_flow_ops.dynamic_stitch([range(input_rank), axes], [input_shape, array_ops.ones(axes_shape, dtype=dtypes.int32)])