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
def all_reduce_indexed_slices(self, input_slices: indexed_slices.IndexedSlices, options: Optional[collective_util.Options]=None) -> indexed_slices.IndexedSlices:
    """All-reduce an IndexedSlices.

    This method can be called outside  tf.function.

    Args:
      input_slices: an IndexedSlices.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced IndexedSlices.
    """
    options = self._options.merge(options)
    with ops.device(self._device):

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
        length = array_ops.shape(input_slices.indices)
        all_lengths = self._all_gather(length, options)

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
        return cond.cond(math_ops.equal(math_ops.reduce_max(all_lengths), math_ops.reduce_min(all_lengths)), lambda: all_gather_indexed_slices(self._all_gather), lambda: all_gather_indexed_slices(all_gather_with_padding))