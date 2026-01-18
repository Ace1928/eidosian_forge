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
def all_gather_indexed_slices(all_gather_fn: Callable[[core.TensorLike, Optional[collective_util.Options]], core.Tensor]) -> indexed_slices.IndexedSlices:
    """Use all_gather_fn to aggregate `IndexedSlices`."""
    all_values = all_gather_fn(input_slices.values, options)
    if options.implementation == collective_util.CommunicationImplementation.NCCL:
        control = [all_values]
    else:
        control = []
    with ops.control_dependencies(control):
        all_indices = all_gather_fn(input_slices.indices, options)
    return indexed_slices.IndexedSlices(values=all_values, indices=all_indices, dense_shape=input_slices.dense_shape)