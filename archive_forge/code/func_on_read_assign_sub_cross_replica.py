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
def on_read_assign_sub_cross_replica(var, value, read_value=True):
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
        if distribute_lib.in_cross_replica_context():
            if var.aggregation == vs.VariableAggregation.SUM:
                raise ValueError('SyncOnReadVariable does not support `assign_sub` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`.')
            return assign_on_each_device(var, assign_sub_on_device, value, read_value)