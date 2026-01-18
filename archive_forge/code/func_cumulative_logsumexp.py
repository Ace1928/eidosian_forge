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
@tf_export('math.cumulative_logsumexp', v1=['math.cumulative_logsumexp'])
@dispatch.add_dispatch_support
def cumulative_logsumexp(x, axis=0, exclusive=False, reverse=False, name=None):
    """Compute the cumulative log-sum-exp of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumulative log-sum-exp, which means
  that the first element of the input is identical to the first element of
  the output.

  This operation is significantly more numerically stable than the equivalent
  tensorflow operation `tf.math.log(tf.math.cumsum(tf.math.exp(x)))`, although
  computes the same result given infinite numerical precision. However, note
  that in some cases, it may be less stable than `tf.math.reduce_logsumexp`
  for a given element, as it applies the "log-sum-exp trick" in a different
  way.

  More precisely, where `tf.math.reduce_logsumexp` uses the following trick:

  ```
  log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)
  ```

  it cannot be directly used here as there is no fast way of applying it
  to each prefix `x[:i]`. Instead, this function implements a prefix
  scan using pairwise log-add-exp, which is a commutative and associative
  (up to floating point precision) operator:

  ```
  log_add_exp(x, y) = log(exp(x) + exp(y))
                    = log(1 + exp(min(x, y) - max(x, y))) + max(x, y)
  ```

  However, reducing using the above operator leads to a different computation
  tree (logs are taken repeatedly instead of only at the end), and the maximum
  is only computed pairwise instead of over the entire prefix. In general, this
  leads to a different and slightly less precise computation.

  Args:
    x: A `Tensor`. Must be one of the following types: `float16`, `float32`,
      `float64`.
    axis: A `Tensor` of type `int32` or `int64` (default: 0). Must be in the
      range `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumulative log-sum-exp.
    reverse: If `True`, performs the cumulative log-sum-exp in the reverse
      direction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same shape and type as `x`.
  """
    with ops.name_scope(name, 'CumulativeLogsumexp', [x]) as name:
        x = ops.convert_to_tensor(x, name='x')
        return gen_math_ops.cumulative_logsumexp(x, axis, exclusive=exclusive, reverse=reverse, name=name)