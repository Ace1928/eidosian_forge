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
def get_samples(self):
    """Returns number of samples in the data, or `None`."""
    if not self.get_size() or not self.batch_size():
        return None
    total_sample = self.get_size() * self.batch_size()
    if self.has_partial_batch():
        total_sample -= self.batch_size() - self.partial_batch_size()
    return total_sample