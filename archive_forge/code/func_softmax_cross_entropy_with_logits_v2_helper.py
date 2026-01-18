import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.softmax_cross_entropy_with_logits_v2'])
@dispatch.add_dispatch_support
@deprecated_args(None, 'dim is deprecated, use axis instead', 'dim')
def softmax_cross_entropy_with_logits_v2_helper(labels, logits, axis=None, name=None, dim=None):
    """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.

  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).

  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Unscaled log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for axis.

  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
    axis = deprecated_argument_lookup('axis', axis, 'dim', dim)
    del dim
    if axis is None:
        axis = -1
    with ops.name_scope(name, 'softmax_cross_entropy_with_logits', [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name='logits')
        labels = ops.convert_to_tensor(labels, name='labels')
        convert_to_float32 = logits.dtype == dtypes.float16 or logits.dtype == dtypes.bfloat16
        precise_logits = math_ops.cast(logits, dtypes.float32) if convert_to_float32 else logits
        labels = math_ops.cast(labels, precise_logits.dtype)
        input_rank = array_ops.rank(precise_logits)
        shape = logits.get_shape()
        if axis != -1:

            def _move_dim_to_end(tensor, dim_index, rank):
                return array_ops.transpose(tensor, array_ops.concat([math_ops.range(dim_index), math_ops.range(dim_index + 1, rank), [dim_index]], 0))
            precise_logits = _move_dim_to_end(precise_logits, axis, input_rank)
            labels = _move_dim_to_end(labels, axis, input_rank)
        input_shape = array_ops.shape(precise_logits)
        precise_logits = _flatten_outer_dims(precise_logits)
        labels = _flatten_outer_dims(labels)
        if config.is_op_determinism_enabled():
            log_probs = log_softmax_v2(precise_logits)
            cost = -math_ops.reduce_sum(labels * log_probs, axis=1)
        else:
            cost, unused_backprop = gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name=name)
        output_shape = array_ops.slice(input_shape, [0], [math_ops.subtract(input_rank, 1)])
        cost = array_ops.reshape(cost, output_shape)
        if not context.executing_eagerly() and shape is not None and (shape.dims is not None):
            shape = shape.as_list()
            del shape[axis]
            cost.set_shape(shape)
        if convert_to_float32:
            return math_ops.cast(cost, logits.dtype)
        else:
            return cost