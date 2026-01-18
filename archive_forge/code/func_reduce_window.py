from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
def reduce_window(operand, init, reducer, window_dimensions, window_strides=None, base_dilations=None, window_dilations=None, padding=None, name=None):
    """Wraps the XLA ReduceWindow operator.

  ReduceWindow is documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    operand: the input tensor
    init: a scalar tensor representing the initial value for the reduction
    reducer: a reduction function that combines a pair of scalars.
    window_dimensions: shape of the window, as a list of integers
    window_strides: inter-window strides, as a list of integers. Optional; if
      omitted, defaults to strides of 1.
    padding: padding to apply to 'operand'. List of (low, high) pairs of
      integers that specify the padding to apply before and after each
      dimension. Optional; if omitted, defaults to no padding.
    name: the operator name, or None.

  Returns:
    A tensor that represents the output of the reduce_window operator.
  """
    window_strides = window_strides or [1] * len(window_dimensions)
    base_dilations = base_dilations or [1] * len(window_dimensions)
    window_dilations = window_dilations or [1] * len(window_dimensions)
    padding = padding or [(0, 0)] * len(window_dimensions)
    return gen_xla_ops.xla_reduce_window(input=operand, init_value=init, window_dimensions=window_dimensions, window_strides=window_strides, base_dilations=base_dilations, window_dilations=window_dilations, padding=padding, computation=reducer, name=name)