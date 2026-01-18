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
def batch_size_for_model_fn(self):
    """Returns the shard batch size for `model_fn`."""
    global_batch_size = self.global_batch_size
    if self.is_running_on_cpu() or (self.is_input_broadcast_with_iterators() and (not self.is_input_slice_broadcast_to_all_cores())):
        return global_batch_size
    return global_batch_size // self.num_replicas