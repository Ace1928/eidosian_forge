from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    conditionally applies gradients if all gradient values are finite.
    Otherwise no update is performed (nor is `global_step` incremented).

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that conditionally applies the specified gradients. If
      `global_step` was not None, that operation also increments `global_step`.

    Raises:
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    if distribute_lib.in_cross_replica_context():
        raise ValueError('apply_gradients() must be called in a replica context.')
    if not self._doing_dynamic_loss_scaling():
        return self._optimizer.apply_gradients(grads_and_vars, global_step, name)
    replica_context = distribute_lib.get_replica_context()
    grads_and_vars = tuple(grads_and_vars)
    return replica_context.merge_call(self._distributed_apply, args=(grads_and_vars, global_step, name))