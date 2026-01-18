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
@contextlib.contextmanager
def catch_stop_iteration(self):
    """Catches errors when an iterator runs out of data."""
    try:
        yield
        self.sync()
    except (StopIteration, errors.OutOfRangeError):
        if self._inferred_steps is None:
            self._inferred_steps = self._current_step
        else:
            self._insufficient_data = True
            total_epochs = self._epochs - self._initial_epoch
            logging.warning('Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, {} batches). You may need to use the repeat() function when building your dataset.'.format(total_epochs * self._inferred_steps))