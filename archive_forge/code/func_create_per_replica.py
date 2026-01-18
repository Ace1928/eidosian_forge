import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def create_per_replica(strategy, value_list):
    """Creates a PerReplica of Tensors from the value_list."""
    if len(strategy.extended.worker_devices) != len(value_list):
        raise ValueError('the length of values must be the same as the number of worker devices')
    tensors = []
    for device, value in zip(strategy.extended.worker_devices, value_list):
        with ops.device(device):
            tensors.append(ops.convert_to_tensor(value))
    return values.PerReplica(tensors)