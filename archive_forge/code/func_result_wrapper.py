from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def result_wrapper(result_fn):
    """Decorator to wrap metric `result()` function in `merge_call()`.

  Result computation is an idempotent operation that simply calculates the
  metric value using the state variables.

  If metric state variables are distributed across replicas/devices and
  `result()` is requested from the context of one device - This function wraps
  `result()` in a distribution strategy `merge_call()`. With this,
  the metric state variables will be aggregated across devices.

  Args:
    result_fn: function that computes the metric result.

  Returns:
    Decorated function that wraps `result_fn()` in distribution strategy
    `merge_call()`.
  """

    def decorated(metric_obj, *args):
        """Decorated function with merge_call."""
        has_strategy = distribute_lib.has_strategy()
        replica_context = distribute_lib.get_replica_context()
        if not has_strategy or replica_context is None or (not distribute_lib.get_strategy().extended._use_merge_call()):
            with distribute_lib.variable_sync_on_read_context():
                raw_result = result_fn(*args)
                if isinstance(raw_result, (tensor.Tensor, variables_module.Variable, float, int)):
                    result_t = array_ops.identity(raw_result)
                elif isinstance(raw_result, dict):
                    result_t = {key: array_ops.identity(value) for key, value in raw_result.items()}
                else:
                    try:
                        result_t = array_ops.identity(raw_result)
                    except (ValueError, TypeError):
                        raise RuntimeError('The output of `metric.result()` can only be a single Tensor/Variable, or a dict of Tensors/Variables. For metric %s, got result %s.' % (metric_obj.name, raw_result))
        else:

            def merge_fn_wrapper(distribution, merge_fn, *args):
                result = distribution.experimental_local_results(merge_fn)[0](*args)
                return array_ops.identity(result)
            result_t = replica_context.merge_call(merge_fn_wrapper, args=(result_fn,) + args)
        metric_obj._call_result = result_t
        return result_t
    return tf_decorator.make_decorator(result_fn, decorated)