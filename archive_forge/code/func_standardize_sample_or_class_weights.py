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
def standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    """Maps `sample_weight` or `class_weight` to model outputs.

  Args:
      x_weight: User-provided `sample_weight` or `class_weight` argument.
      output_names: List of output names (strings) in the model.
      weight_type: A string used purely for exception printing.

  Returns:
      A list of `sample_weight` or `class_weight` where there are exactly
          one element per model output.

  Raises:
      ValueError: In case of invalid user-provided argument.
  """
    if x_weight is None or (isinstance(x_weight, (list, tuple)) and len(x_weight) == 0):
        return [None for _ in output_names]
    if len(output_names) == 1:
        if isinstance(x_weight, (list, tuple)) and len(x_weight) == 1:
            return x_weight
        if isinstance(x_weight, dict) and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if isinstance(x_weight, (list, tuple)):
        if len(x_weight) != len(output_names):
            raise ValueError('Provided `' + weight_type + '` was a list of ' + str(len(x_weight)) + ' elements, but the model has ' + str(len(output_names)) + ' outputs. You should provide one `' + weight_type + '`array per model output.')
        return x_weight
    if isinstance(x_weight, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys(weight_type, x_weight, output_names)
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise TypeError('The model has multiple outputs, so `' + weight_type + '` should be either a list or a dict. Provided `' + weight_type + '` type not understood: ' + str(x_weight))