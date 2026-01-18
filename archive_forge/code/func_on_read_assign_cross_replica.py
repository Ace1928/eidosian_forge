from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object
def on_read_assign_cross_replica(var, value, read_value=True):
    """Return the value of the variable in cross replica context."""
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
        if distribute_lib.in_cross_replica_context():
            tensor = value
            if var.aggregation == vs.VariableAggregation.SUM:
                strategy = var._distribute_strategy
                tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync, var.dtype)
            return assign_on_each_device(var, assign_on_device, tensor, read_value)