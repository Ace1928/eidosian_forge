import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def stitch_values(values_and_indices_list):
    """Stitch values together according to their indices.

  Args:
    values_and_indices_list: a list of tuples of values and indices indicating
      the values and positions in the returned list.

  Returns:
    a stitched list of values.
  """
    length = 0
    for values_and_indices in values_and_indices_list:
        length += len(values_and_indices[0])
    result = [None] * length
    for values_and_indices in values_and_indices_list:
        if values_and_indices and values_and_indices[0]:
            for v, i in zip(*values_and_indices):
                assert result[i] is None
                result[i] = v
    return result