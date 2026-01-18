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
def divide_by_n_tensors_or_indexed_slices(value, n):
    if isinstance(value, indexed_slices.IndexedSlices):
        value = backprop_util.FlattenNestedIndexedSlices(value)
        return indexed_slices.IndexedSlices(value.values / n, value.indices, value.dense_shape)
    else:
        return value / n