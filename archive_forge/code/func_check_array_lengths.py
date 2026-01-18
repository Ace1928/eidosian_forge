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
def check_array_lengths(inputs, targets, weights=None):
    """Does user input validation for numpy arrays.

  Args:
      inputs: list of Numpy arrays of inputs.
      targets: list of Numpy arrays of targets.
      weights: list of Numpy arrays of sample weights.

  Raises:
      ValueError: in case of incorrectly formatted data.
  """

    def is_tensor_or_composite_tensor(x):
        return tensor_util.is_tf_type(x) or is_composite_or_composite_value(x)

    def set_of_lengths(x):
        if x is None:
            return {}
        else:
            return set([y.shape[0] for y in x if y is not None and (not is_tensor_or_composite_tensor(y))])
    set_x = set_of_lengths(inputs)
    set_y = set_of_lengths(targets)
    set_w = set_of_lengths(weights)
    if len(set_x) > 1:
        raise ValueError('All input arrays (x) should have the same number of samples. Got array shapes: ' + str([x.shape for x in inputs]))
    if len(set_y) > 1:
        raise ValueError('All target arrays (y) should have the same number of samples. Got array shapes: ' + str([y.shape for y in targets]))
    if set_x and set_y and (list(set_x)[0] != list(set_y)[0]):
        raise ValueError('Input arrays should have the same number of samples as target arrays. Found ' + str(list(set_x)[0]) + ' input samples and ' + str(list(set_y)[0]) + ' target samples.')
    if len(set_w) > 1:
        raise ValueError('All sample_weight arrays should have the same number of samples. Got array shapes: ' + str([w.shape for w in weights]))
    if set_y and set_w and (list(set_y)[0] != list(set_w)[0]):
        raise ValueError('Sample_weight arrays should have the same number of samples as target arrays. Got ' + str(list(set_y)[0]) + ' input samples and ' + str(list(set_w)[0]) + ' target samples.')