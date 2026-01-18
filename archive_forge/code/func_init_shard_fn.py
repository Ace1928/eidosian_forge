import functools
import os
import threading
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as base_cluster_resolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import server_lib
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def init_shard_fn(shard_index):
    if not init_from_fn:
        logging.log_if(logging.WARN, _INEFFICIENT_INIT_WARNING % name, shard_index == 0 and shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
        return initial_value[offsets[shard_index]:offsets[shard_index + 1]]
    partition_shape = (offsets[shard_index + 1] - offsets[shard_index],) + shape[1:]
    partition_offset = (offsets[shard_index],) + (0,) * len(shape[1:])
    arg_spec = tf_inspect.getfullargspec(initial_value)
    if 'shard_info' not in arg_spec.args and 'shard_info' not in arg_spec.kwonlyargs:
        try:
            value = initial_value(partition_shape=partition_shape, partition_offset=partition_offset)
        except (TypeError, ValueError):
            value = initial_value()
        if value.shape == partition_shape:
            return value
        else:
            logging.log_if(logging.WARN, _INEFFICIENT_INIT_WARNING % name, shard_index == 0 and shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
            return value[offsets[shard_index]:offsets[shard_index + 1]]
    else:
        return initial_value(shard_info=trackable.ShardInfo(shape=tensor_shape.as_shape(partition_shape), offset=partition_offset))