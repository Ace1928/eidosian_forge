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
def copy_tensor_or_indexed_slices_to_device(value, device):
    """Copies a tensor or IndexedSlices to a device."""
    with ops.device(device):
        if isinstance(value, indexed_slices.IndexedSlices):
            copied_values = array_ops.identity(value.values)
            copied_indices = array_ops.identity(value.indices)
            if value.dense_shape is not None:
                copied_shape = array_ops.identity(value.dense_shape)
            else:
                copied_shape = None
            result = indexed_slices.IndexedSlices(copied_values, copied_indices, copied_shape)
        else:
            result = array_ops.identity(value)
    return result