import functools
import math
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def convert_to_generator_like(data, batch_size=None, steps_per_epoch=None, epochs=1, shuffle=False):
    """Make a generator out of NumPy or EagerTensor inputs.

  Args:
    data: Either a generator or `keras.utils.data_utils.Sequence` object or
      `Dataset`, `Iterator`, or a {1,2,3}-tuple of NumPy arrays or EagerTensors.
      If a tuple, the elements represent `(x, y, sample_weights)` and may be
      `None` or `[None]`.
    batch_size: Used when creating a generator out of tuples of NumPy arrays or
      EagerTensors.
    steps_per_epoch: Steps of the generator to run each epoch. If `None` the
      number of steps will be read from the data (for
      `keras.utils.data_utils.Sequence` types).
    epochs: Total number of epochs to run.
    shuffle: Whether the data should be shuffled.

  Returns:
    - Generator, `keras.utils.data_utils.Sequence`, or `Iterator`.

  Raises:
    - ValueError: If `batch_size` is not provided for NumPy or EagerTensor
      inputs.
  """
    if isinstance(data, tuple):
        data = tuple((ele for ele in data if not all((e is None for e in nest.flatten(ele)))))
    if data_utils.is_generator_or_sequence(data) or isinstance(data, iterator_ops.IteratorBase):
        if isinstance(data, data_utils.Sequence):
            if steps_per_epoch is None:
                steps_per_epoch = len(data)
        return (data, steps_per_epoch)
    if isinstance(data, data_types.DatasetV2):
        return (dataset_ops.make_one_shot_iterator(data), steps_per_epoch)
    num_samples = int(nest.flatten(data)[0].shape[0])
    if batch_size is None:
        raise ValueError('When passing input data as arrays, do not specify `steps_per_epoch`/`steps` argument. Please use `batch_size` instead.')
    steps_per_epoch = int(math.ceil(num_samples / batch_size))

    def _gen(data):
        """Makes a generator out of a structure of NumPy/EagerTensors."""
        index_array = np.arange(num_samples)
        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(index_array)
            batches = generic_utils.make_batches(num_samples, batch_size)
            for batch_start, batch_end in batches:
                batch_ids = index_array[batch_start:batch_end]
                flat_batch_data = training_utils.slice_arrays(nest.flatten(data), batch_ids, contiguous=not shuffle)
                yield nest.pack_sequence_as(data, flat_batch_data)
    return (_gen(data), steps_per_epoch)