from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
@property
def batch_size_for_input_fn(self):
    """Returns the shard batch size for `input_fn`."""
    global_batch_size = self.global_batch_size
    if self.is_running_on_cpu() or self.is_input_broadcast_with_iterators():
        return global_batch_size
    if self.is_input_sharded_per_core() or self.is_input_per_host_with_iterators() or self.is_replica_across_hosts():
        return global_batch_size // self.num_replicas
    else:
        return global_batch_size // self.num_hosts