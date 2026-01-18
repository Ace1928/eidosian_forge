import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def slice_batch_indices(indices):
    """Convert a Tensor of indices into a dataset of batched indices.

      This step can be accomplished in several ways. The most natural is to
      slice the Tensor in a Dataset map. (With a condition on the upper index to
      handle the partial batch.) However it turns out that coercing the Tensor
      into a shape which is divisible by the batch size (and handling the last
      partial batch separately) allows for a much more favorable memory access
      pattern and improved performance.

      Args:
        indices: Tensor which determines the data order for an entire epoch.

      Returns:
        A Dataset of batched indices.
      """
    num_in_full_batch = num_full_batches * batch_size
    first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
    first_k_indices = array_ops.reshape(first_k_indices, [num_full_batches, batch_size])
    flat_dataset = dataset_ops.DatasetV2.from_tensor_slices(first_k_indices)
    if self._partial_batch_size:
        index_remainder = dataset_ops.DatasetV2.from_tensors(array_ops.slice(indices, [num_in_full_batch], [self._partial_batch_size]))
        flat_dataset = flat_dataset.concatenate(index_remainder)
    if shuffle == 'batch':
        flat_dataset = flat_dataset.shuffle(1024).repeat(epochs)
    return flat_dataset