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
def steps(self):
    """Yields steps for the current epoch."""
    self._current_step = 0
    while self._inferred_steps is None or self._current_step < self._inferred_steps:
        if self._insufficient_data:
            break
        can_run_full_execution = self._steps_per_execution_value == 1 or self._inferred_steps is None or self._inferred_steps - self._current_step >= self._steps_per_execution_value
        if can_run_full_execution:
            self._step_increment = self._steps_per_execution_value - 1
            yield self._current_step
            self._current_step += self._steps_per_execution_value
        else:
            steps_remaining = self._inferred_steps - self._current_step
            self._steps_per_execution.assign(steps_remaining)
            self._step_increment = steps_remaining - 1
            yield self._current_step
            self._current_step += steps_remaining
            self._steps_per_execution.assign(self._steps_per_execution_value)