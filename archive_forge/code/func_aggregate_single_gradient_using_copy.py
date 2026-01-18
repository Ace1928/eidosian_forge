import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def aggregate_single_gradient_using_copy(grad_and_vars, use_mean, check_inf_nan):
    """Calculate the average gradient for a shared variable across all replicas.

  Note that this function provides a synchronization point across all replicas.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single replica, and the number of pairs
      equals the number of replicas.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all replicas. The variable is chosen
      from the first replica. The has_nan_or_inf indicates the grads has nan or
      inf.
  """
    grads = [g for g, _ in grad_and_vars]
    grad = math_ops.add_n(grads)
    if use_mean and len(grads) > 1:
        grad = array_ops.multiply(grad, 1.0 / len(grads))
    v = grad_and_vars[0][1]
    if check_inf_nan:
        has_nan_or_inf = array_ops.logical_not(array_ops.reduce_all(array_ops.is_finite(grads)))
        return ((grad, v), has_nan_or_inf)
    else:
        return ((grad, v), None)