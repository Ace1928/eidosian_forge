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
def aggregate_tensors_or_indexed_slices(values, accumulation_fn=math_ops.add_n):
    """Aggregate tensors using `accumulation_fn` and IndexedSlices via concat."""
    if any((isinstance(v, indexed_slices.IndexedSlices) for v in values)):
        return backprop_util.AggregateIndexedSlicesGradients(values)
    else:
        return accumulation_fn(values)