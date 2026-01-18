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
def all_gather_with_padding(input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
    """all_gather tensors of different sizes using padding."""
    max_length = math_ops.reduce_max(all_lengths)
    padded_tensor = _pad_util(input_tensor, max_length)
    all_padded_tensors = self._all_gather(padded_tensor, options)
    split_tensors = []
    for i in range(self._group_size):
        start_pos = i * max_length
        split_tensors.append(all_padded_tensors[start_pos:start_pos + all_lengths[i]])
    return array_ops.concat(split_tensors, 0)