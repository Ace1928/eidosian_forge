import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def prepare_loss_weights(training_endpoints, loss_weights=None):
    """Converts loss weights to a list of loss weights.

  The result loss weights will be populated on the training endpoint.

  Args:
      training_endpoints: List of model training endpoints.
      loss_weights: Optional list or dictionary specifying scalar coefficients
        (Python floats) to weight the loss contributions of different model
        outputs. The loss value that will be minimized by the model will then be
        the *weighted sum* of all individual losses, weighted by the
          `loss_weights` coefficients. If a list, it is expected to have a 1:1
            mapping to the model's outputs. If a dict, it is expected to map
            output names (strings) to scalar coefficients.

  Raises:
      ValueError: If loss weight is a dict with key not in model output names,
          or if loss is a list with len not equal to model outputs.
  """
    if loss_weights is None:
        for e in training_endpoints:
            e.loss_weight = 1.0
    elif isinstance(loss_weights, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('loss_weights', loss_weights, [e.output_name for e in training_endpoints])
        for e in training_endpoints:
            e.loss_weight = loss_weights.get(e.output_name, 1.0)
    elif isinstance(loss_weights, list):
        if len(loss_weights) != len(training_endpoints):
            raise ValueError('When passing a list as loss_weights, it should have one entry per model output. The model has ' + str(len(training_endpoints)) + ' outputs, but you passed loss_weights=' + str(loss_weights))
        for w, e in zip(loss_weights, training_endpoints):
            e.loss_weight = w
    else:
        raise TypeError('Could not interpret loss_weights argument: ' + str(loss_weights) + ' - expected a list of dicts.')