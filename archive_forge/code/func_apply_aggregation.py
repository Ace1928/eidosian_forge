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
def apply_aggregation(strategy, value, aggregation, destinations):
    if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
        return strategy.extended.broadcast_to(strategy.experimental_local_results(value)[0], destinations=destinations)
    reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
    return strategy.extended.reduce_to(reduce_op, value, destinations)