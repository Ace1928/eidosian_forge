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
def should_run_validation(validation_freq, epoch):
    """Checks if validation should be run this epoch.

  Args:
    validation_freq: Integer or list. If an integer, specifies how many training
      epochs to run before a new validation run is performed. If a list,
      specifies the epochs on which to run validation.
    epoch: Integer, the number of the training epoch just completed.

  Returns:
    Bool, True if validation should be run.

  Raises:
    ValueError: if `validation_freq` is an Integer and less than 1, or if
    it is neither an Integer nor a Sequence.
  """
    one_indexed_epoch = epoch + 1
    if isinstance(validation_freq, int):
        if validation_freq < 1:
            raise ValueError('`validation_freq` can not be less than 1.')
        return one_indexed_epoch % validation_freq == 0
    if not isinstance(validation_freq, collections.abc.Container):
        raise ValueError('`validation_freq` must be an Integer or `collections.abc.Container` (e.g. list, tuple, etc.)')
    return one_indexed_epoch in validation_freq