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
def standardize_input_data(data, names, shapes=None, check_batch_axis=True, exception_prefix=''):
    """Normalizes inputs and targets provided by users.

  Users may pass data as a list of arrays, dictionary of arrays,
  or as a single array. We normalize this to an ordered list of
  arrays (same order as `names`), while checking that the provided
  arrays have shapes that match the network's expectations.

  Args:
      data: User-provided input data (polymorphic).
      names: List of expected array names.
      shapes: Optional list of expected array shapes.
      check_batch_axis: Boolean; whether to check that the batch axis of the
        arrays matches the expected value found in `shapes`.
      exception_prefix: String prefix used for exception formatting.

  Returns:
      List of standardized input arrays (one array per model input).

  Raises:
      ValueError: in case of improperly formatted user-provided data.
  """
    try:
        data_len = len(data)
    except TypeError:
        data_len = None
    if not names:
        if data_len and (not isinstance(data, dict)):
            raise ValueError('Error when checking model ' + exception_prefix + ': expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]
    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError('No data provided for "' + e.args[0] + '". Need data for each key in: ' + str(names))
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], (list, tuple)):
            data = [np.asarray(d) for d in data]
        elif len(names) == 1 and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    if shapes is not None:
        data = [standardize_single_array(x, shape) for x, shape in zip(data, shapes)]
    else:
        data = [standardize_single_array(x) for x in data]
    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError('Error when checking model ' + exception_prefix + ': the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see ' + str(len(names)) + ' array(s), ' + 'for inputs ' + str(names) + ' but instead got the following list of ' + str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError('Error when checking model ' + exception_prefix + ': you are passing a list as input to your model, but the model expects a list of ' + str(len(names)) + ' Numpy arrays instead. The list you passed was: ' + str(data)[:200])
        elif len(data) == 1 and (not hasattr(data[0], 'shape')):
            raise TypeError('Error when checking model ' + exception_prefix + ': data should be a Numpy array, or list/dict of Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                if tensor_util.is_tf_type(data[i]):
                    tensorshape = data[i].shape
                    if not tensorshape:
                        continue
                    data_shape = tuple(tensorshape.as_list())
                elif is_composite_or_composite_value(data[i]):
                    tensorshape = get_composite_shape(data[i])
                    data_shape = tuple(tensorshape.as_list())
                else:
                    data_shape = data[i].shape
                shape = shapes[i]
                if len(data_shape) != len(shape):
                    raise ValueError('Error when checking ' + exception_prefix + ': expected ' + names[i] + ' to have ' + str(len(shape)) + ' dimensions, but got array with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim is not None and (dim is not None):
                        raise ValueError('Error when checking ' + exception_prefix + ': expected ' + names[i] + ' to have shape ' + str(shape) + ' but got array with shape ' + str(data_shape))
    return data