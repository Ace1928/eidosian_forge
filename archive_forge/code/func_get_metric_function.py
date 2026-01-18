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
def get_metric_function(metric, output_shape=None, loss_fn=None):
    """Returns the metric function corresponding to the given metric input.

  Args:
      metric: Metric function name or reference.
      output_shape: The shape of the output that this metric will be calculated
        for.
      loss_fn: The loss function used.

  Returns:
      The metric function.
  """
    if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
        return metrics_module.get(metric)
    is_sparse_categorical_crossentropy = isinstance(loss_fn, losses.SparseCategoricalCrossentropy) or (isinstance(loss_fn, losses.LossFunctionWrapper) and loss_fn.fn == losses.sparse_categorical_crossentropy)
    is_binary_crossentropy = isinstance(loss_fn, losses.BinaryCrossentropy) or (isinstance(loss_fn, losses.LossFunctionWrapper) and loss_fn.fn == losses.binary_crossentropy)
    if metric in ['accuracy', 'acc']:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_accuracy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_accuracy
        return metrics_module.categorical_accuracy
    else:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_crossentropy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_crossentropy
        return metrics_module.categorical_crossentropy