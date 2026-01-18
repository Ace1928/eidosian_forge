import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('quantization.quantize_and_dequantize_v2')
@dispatch.add_dispatch_support
def quantize_and_dequantize_v2(input, input_min, input_max, signed_input=True, num_bits=8, range_given=False, round_mode='HALF_TO_EVEN', name=None, narrow_range=False, axis=None):
    """Quantizes then dequantizes a tensor.

  Updates the gradient definition for quantization that is outside the range to
  be 0.To simulate the V1 the behavior of
  tf.quantization.quantize_and_dequantize(...) use
  tf.grad_pass_through(tf.quantization.quantize_and_dequantize_v2)(...).

  Example usage:

  ```python
  def getQuantizeOp(input):
      input_tensor = tf.placeholder(tf.float32, shape=[4, 4])
      net = tf.quantization.quantize_and_dequantize(input,
                                                    input_min=min_threshold,
                                                    input_max=max_threshold,
                                                    range_given=True)

  To simulate v1 behavior:

  def testDecomposeQuantizeDequantize(self):
      def f(input_tensor):
        return tf.quantization.quantize_and_dequantize_v2(input_tensor,
                                                          input_min = 5.0,
                                                          input_max= -10.0,
                                                          range_given=True)
      input_tensor = tf.placeholder(tf.float32, shape=[4, 4])
      net = tf.grad_pass_through(f)(input_tensor)
  ```

  Args:
    input: A `Tensor` to quantize and dequantize.
    input_min: If range_given=True, the minimum input value, that needs to be
      represented in the quantized representation. If axis is specified, this
      should be a vector of minimum values for each slice along axis.
    input_max: If range_given=True, the maximum input value that needs to be
      represented in the quantized representation. If axis is specified, this
      should be a vector of maximum values for each slice along axis.
    signed_input: True if the quantization is signed or unsigned.
    num_bits: The bitwidth of the quantization.
    range_given: If true use `input_min` and `input_max` for the range of the
      input, otherwise determine min and max from the input `Tensor`.
    round_mode: Rounding mode when rounding from float values to quantized ones.
      one of ['HALF_TO_EVEN', 'HALF_UP']
    name: Optional name for the operation.
    narrow_range: If true, then the absolute value of the quantized minimum
      value is the same as the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    axis: Integer. If specified, refers to a dimension of the input tensor, such
      that quantization will be per slice along that dimension.

  Returns:
    A `Tensor`. Each element is the result of quantizing and dequantizing the
    corresponding element of `input`.
  """
    if axis is None:
        axis = -1
    elif axis < 0:
        if input.shape.ndims is None:
            raise ValueError('input should have known rank to use negative axis.')
        axis %= input.shape.ndims
    return gen_array_ops.quantize_and_dequantize_v4(input, input_min=input_min, input_max=input_max, signed_input=signed_input, num_bits=num_bits, range_given=range_given, round_mode=round_mode, narrow_range=narrow_range, axis=axis, name=name)