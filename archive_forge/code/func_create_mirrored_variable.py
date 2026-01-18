from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def create_mirrored_variable(strategy, real_mirrored_creator, class_mapping, policy_mapping, **kwargs):
    """Create distributed variables with given synchronization and aggregation."""
    if kwargs.pop('experimental_batch_initialization', None):
        variable_class_key = 'LazyVariableClass'
    else:
        variable_class_key = 'VariableClass'
    var_collections = kwargs.pop('collections', None)
    if var_collections is None:
        var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    kwargs['collections'] = []
    synchronization = _validate_synchronization(kwargs)
    kwargs['synchronization'] = synchronization
    aggregation = _validate_aggregation(kwargs)
    use_var_policy = getattr(strategy.extended, '_use_var_policy', False)
    kwargs.pop('caching_device', None)
    with record.stop_recording():
        value_list = real_mirrored_creator(**kwargs)
        for v in value_list:
            if hasattr(v, '_initializer_op') and v._initializer_op is None:
                v._initializer_op = control_flow_ops.no_op()
        if use_var_policy:
            var_policy_cls = policy_mapping.get(synchronization)
            var_policy = var_policy_cls(aggregation=aggregation)
            var_cls = class_mapping.get(variable_class_key)
            result = var_cls(strategy, value_list, aggregation, var_policy=var_policy)
        else:
            var_cls = class_mapping.get(synchronization)
            result = var_cls(strategy, value_list, aggregation)
    if not context.executing_eagerly():
        g = ops.get_default_graph()
        if kwargs.get('trainable', True):
            var_collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
            l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
            for value in value_list:
                for i, trainable_variable in enumerate(l):
                    if value is trainable_variable:
                        del l[i]
                        break
        g.add_to_collections(var_collections, result)
    elif ops.GraphKeys.GLOBAL_STEP in var_collections:
        ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)
    return result