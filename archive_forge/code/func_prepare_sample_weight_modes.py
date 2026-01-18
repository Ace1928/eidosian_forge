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
def prepare_sample_weight_modes(training_endpoints, sample_weight_mode):
    """Prepares sample weight modes for the model.

  Args:
    training_endpoints: List of model _TrainingEndpoints.
    sample_weight_mode: sample weight mode user input passed from compile API.

  Raises:
    ValueError: In case of invalid `sample_weight_mode` input.
  """
    if isinstance(sample_weight_mode, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('sample_weight_mode', sample_weight_mode, [e.output_name for e in training_endpoints])
        for end_point in training_endpoints:
            if not end_point.should_skip_target_weights():
                if end_point.output_name not in sample_weight_mode:
                    raise ValueError('Output ' + end_point.output_name + 'missing from `_sample_weight_modes` dictionary')
                else:
                    end_point.sample_weight_mode = sample_weight_mode.get(end_point.output_name)
    elif isinstance(sample_weight_mode, (list, tuple)):
        if len(sample_weight_mode) != len(training_endpoints):
            raise ValueError('When passing a list as sample_weight_mode, it should have one entry per model output. The model has ' + str(len(training_endpoints)) + ' outputs, but you passed ' + str(len(sample_weight_mode)) + '_sample_weight_modes.')
        for mode, endpoint in zip(sample_weight_mode, training_endpoints):
            if not endpoint.should_skip_target_weights():
                endpoint.sample_weight_mode = mode
    else:
        for endpoint in training_endpoints:
            if not endpoint.should_skip_target_weights():
                endpoint.sample_weight_mode = sample_weight_mode