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
def process_batch_and_step_size(strategy, inputs, batch_size, steps_per_epoch, mode, validation_split=0.0):
    """Process the batch size and step size based on input and dist strategy."""
    first_x_value = nest.flatten(inputs)[0]
    if isinstance(first_x_value, np.ndarray):
        num_samples = first_x_value.shape[0]
        if validation_split and 0.0 < validation_split < 1.0:
            num_samples = int(num_samples * (1 - validation_split))
        steps_per_epoch, batch_size = get_input_params(strategy, num_samples, steps_per_epoch, batch_size, mode=mode)
    return (batch_size, steps_per_epoch)