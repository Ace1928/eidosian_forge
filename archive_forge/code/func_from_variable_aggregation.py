import enum
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def from_variable_aggregation(aggregation):
    mapping = {variable_scope.VariableAggregation.SUM: ReduceOp.SUM, variable_scope.VariableAggregation.MEAN: ReduceOp.MEAN}
    reduce_op = mapping.get(aggregation)
    if not reduce_op:
        raise ValueError('Could not convert from `tf.VariableAggregation` %s to`tf.distribute.ReduceOp` type' % aggregation)
    return reduce_op