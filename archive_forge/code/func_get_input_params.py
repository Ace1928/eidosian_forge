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
def get_input_params(distribution_strategy, num_samples, steps, batch_size, mode=None):
    """Calculate the number of batches and steps/steps_per_epoch.

  Args:
    distribution_strategy: The DistributionStrategy used to compile the model.
    num_samples: The number of samples from which we determine the batch size
      and steps.
    steps:  The specified number of steps.
    batch_size: The specified batch_size.
    mode: ModeKey representing whether input will be used for training,
      evaluation, or prediction. This is used to relax the constraints on
      consuming all the training samples to keep compatibility till we support
      partial batches. If none, then partial batches are not allowed.

  Returns:
    steps: The steps or steps_per_epoch argument depending on if a user is
        calling `fit`, `evaluate` or `predict`. If the is_training flag is set
        we don't require the number of samples to be used completely.
    batch_size: The batch size to be used in model iterations.

  Raises:
    ValueError: If the number of batches or steps evaluates to 0.

  """
    use_per_replica_batch = not dist_utils.global_batch_size_supported(distribution_strategy)
    if context.executing_eagerly():
        allow_partial_batch = mode != ModeKeys.TRAIN or not backend.is_tpu_strategy(distribution_strategy)
    else:
        allow_partial_batch = mode == ModeKeys.TRAIN or ((mode == ModeKeys.PREDICT or mode == ModeKeys.TEST) and backend.is_tpu_strategy(distribution_strategy))
    if steps is None:
        if batch_size is None:
            global_batch_size = min(num_samples, 32)
        else:
            global_batch_size = batch_size
            if use_per_replica_batch:
                global_batch_size *= distribution_strategy.num_replicas_in_sync
        if allow_partial_batch:
            steps = np.ceil(num_samples / global_batch_size).astype(int)
        else:
            if num_samples % global_batch_size:
                raise ValueError('The number of samples %s is not divisible by batch size %s.' % (num_samples, global_batch_size))
            steps = num_samples // global_batch_size
    elif batch_size is None:
        if num_samples % steps:
            raise ValueError('The number of samples %s is not divisible by steps %s. Please change the number of steps to a value that can consume all the samples' % (num_samples, steps))
        global_batch_size = num_samples // steps
    else:
        global_batch_size = batch_size
        if use_per_replica_batch:
            global_batch_size *= distribution_strategy.num_replicas_in_sync
        min_num_samples = global_batch_size * steps
        if allow_partial_batch:
            min_num_samples = global_batch_size * (steps - 1) + 1 if steps > 1 else 0
        if num_samples < min_num_samples:
            raise ValueError('Number of samples %s is less than samples required for specified batch_size %s and steps %s' % (num_samples, global_batch_size, steps))
    if use_per_replica_batch:
        if global_batch_size % distribution_strategy.num_replicas_in_sync:
            raise ValueError('The batch size (%s) could not be sharded evenly across the sync replicas (%s) in the distribution strategy.' % (global_batch_size, distribution_strategy.num_replicas_in_sync))
        batch_size = global_batch_size // distribution_strategy.num_replicas_in_sync
    else:
        batch_size = global_batch_size
    return (steps, batch_size)