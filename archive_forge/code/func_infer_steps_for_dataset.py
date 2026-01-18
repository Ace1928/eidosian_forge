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
def infer_steps_for_dataset(model, dataset, steps, epochs=1, steps_name='steps'):
    """Infers steps_per_epoch needed to loop through a dataset.

  Args:
      model: Keras model instance.
      dataset: Input data of type tf.data.Dataset.
      steps: Number of steps to draw from the dataset (may be None if unknown).
      epochs: Number of times to iterate over the dataset.
      steps_name: The string name of the steps argument, either `steps`,
        `validation_steps`, or `steps_per_epoch`. Only used for error message
        formatting.

  Returns:
    Integer or `None`. Inferred number of steps to loop through the dataset.
    `None` is returned if 1) the size of the dataset is unknown and `steps` was
    not specified, or 2) this is multi-worker training and auto sharding is
    enabled.

  Raises:
    ValueError: In case of invalid argument values.
  """
    assert isinstance(dataset, data_types.DatasetV2)
    if model._in_multi_worker_mode() and dataset.options().experimental_distribute.auto_shard_policy != options_lib.AutoShardPolicy.OFF:
        return None
    size = backend.get_value(cardinality.cardinality(dataset))
    if size == cardinality.INFINITE and steps is None:
        raise ValueError('When passing an infinitely repeating dataset, you must specify the `%s` argument.' % (steps_name,))
    if size >= 0:
        if steps is not None and steps * epochs > size:
            if epochs > 1:
                raise ValueError('The dataset you passed contains %s batches, but you passed `epochs=%s` and `%s=%s`, which is a total of %s steps. We cannot draw that many steps from this dataset. We suggest to set `%s=%s`.' % (size, epochs, steps_name, steps, steps * epochs, steps_name, size // epochs))
            else:
                raise ValueError('The dataset you passed contains %s batches, but you passed `%s=%s`. We cannot draw that many steps from this dataset. We suggest to set `%s=%s`.' % (size, steps_name, steps, steps_name, size))
    if steps is None:
        if size >= 0:
            return size
        return None
    return steps