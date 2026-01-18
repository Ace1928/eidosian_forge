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
@tf_export('math.special.fresnel_cos')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def fresnel_cos(x, name=None):
    """Computes Fresnel's cosine integral of `x` element-wise.

  The Fresnel cosine integral is defined as the integral of `cos(t^2)` from
  `0` to `x`, with the domain of definition all real numbers.

  The Fresnel cosine integral is odd.
  >>> tf.math.special.fresnel_cos([-1., -0.1, 0.1, 1.]).numpy()
  array([-0.7798934 , -0.09999753,  0.09999753,  0.7798934 ], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.fresnel second output.
  @end_compatibility
  """
    with ops.name_scope(name, 'fresnel_cos', [x]):
        return gen_special_math_ops.fresnel_cos(x)