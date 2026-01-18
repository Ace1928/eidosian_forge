import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.special.expint')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def expint(x, name=None):
    """Computes the Exponential integral of `x` element-wise.

  The Exponential integral is defined as the integral of `exp(t) / t` from
  `-inf` to `x`, with the domain of definition all positive real numbers.

  >>> tf.math.special.expint([1., 1.1, 2.1, 4.1]).numpy()
  array([ 1.8951179,  2.1673784,  5.3332353, 21.048464], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.expi
  @end_compatibility
  """
    with ops.name_scope(name, 'expint', [x]):
        return gen_special_math_ops.expint(x)