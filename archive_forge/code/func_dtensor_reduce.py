from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
def dtensor_reduce(strategy, reduce_op, value, axis):
    """Implement dtensor based strategy.reduce()."""
    distribute_lib._require_cross_replica_or_default_context_extended(strategy.extended)
    if isinstance(reduce_op, str):
        reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    distributed_input = is_distributed_value(value)
    if not distributed_input and axis is None:
        destinations = device_util.current() or strategy.extended._default_device or '/device:CPU:0'
        devices = cross_device_ops_lib.get_devices_from(destinations)
        with ops.device(devices[0]):
            return array_ops.identity(cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, strategy.num_replicas_in_sync))
    value = convert_inputs_to_dtensor(value, strategy._mesh)
    if reduce_op == reduce_util.ReduceOp.MEAN:
        reduce_op = math_ops.reduce_mean
    else:
        reduce_op = math_ops.reduce_sum
    if d_api.fetch_layout(value).is_fully_replicated():
        if axis is not None:
            value = reduce_op(value, axis=axis)
    else:
        new_shape = [strategy.num_replicas_in_sync, -1]
        if len(value.shape) > 1:
            new_shape.extend(array_ops.shape(value)[1:])
        value = array_ops.reshape(value, new_shape)
        if axis is not None:
            value = reduce_op(value, axis=axis + 1)
        value = reduce_op(value, axis=0)
    return value