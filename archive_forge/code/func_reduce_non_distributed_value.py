import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def reduce_non_distributed_value(reduce_op, value, destinations, num_replicas_in_graph, canonicalize_devices=True):
    """Reduce a non-DistributedValue `value` to `destinations`."""
    if isinstance(value, value_lib.DistributedValues):
        raise ValueError('You are passing a `DistributedValues` to `reduce_non_distributed_value`, which is not allowed.')
    if not tensor_util.is_tf_type(value) and np.all(value == 0):
        return np.zeros(value.shape, dtype=value.dtype)
    if reduce_op == reduce_util.ReduceOp.MEAN:
        return value
    elif num_replicas_in_graph != 1:
        raise ValueError('A non-DistributedValues value %s cannot be reduced with the given reduce op %s.' % (value, reduce_op))
    else:
        validate_destinations(destinations)
        return simple_broadcast(value, destinations, canonicalize_devices=canonicalize_devices)