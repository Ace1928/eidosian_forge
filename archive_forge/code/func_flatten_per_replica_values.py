import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def flatten_per_replica_values(distribution_strategy, per_replica_values):
    """Unwraps and flattens a nest of PerReplica parameters.

  PerReplica values have one value associated with each device. Each entry in
  the PerReplica dict has a device `key` and the corresponding value on the
  device as the `value`. In this function we take a PerReplica value or a list
  of PerReplica values and return all the values in the PerReplica dict.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
      validation.
    per_replica_values: List of PerReplica object or a single PerReplica object.

  Returns:
    List of values of all the PerReplica objects.

  """
    return [e for flattened in nest.flatten(per_replica_values) for e in distribution_strategy.unwrap(flattened)]