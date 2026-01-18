import copy
import threading
import time
import weakref
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def initial_value_fn():
    group_key = self._collective_keys.get_group_key([device])
    group_size = self._num_workers
    collective_instance_key = self._collective_keys.get_instance_key(group_key, device)
    with ops.device(device):
        initial_value = kwargs['initial_value']
        if callable(initial_value):
            initial_value = initial_value()
        if isinstance(initial_value, base.CheckpointInitialValue):
            initial_value = initial_value.wrapped_value
        assert not callable(initial_value)
        initial_value = ops.convert_to_tensor(initial_value, dtype=kwargs.get('dtype', None))
        if self._num_workers > 1:
            if self._is_chief:
                bcast_send = collective_ops.broadcast_send(initial_value, initial_value.shape, initial_value.dtype, group_size, group_key, collective_instance_key)
                with ops.control_dependencies([bcast_send]):
                    return array_ops.identity(initial_value)
            else:
                return collective_ops.broadcast_recv(initial_value.shape, initial_value.dtype, group_size, group_key, collective_instance_key)
        return initial_value