import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def set_configuration_from_sharded_input_tensors(self, input_tensors):
    """Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of lists of Tensors whose types and shapes are used
    to set the queue configuration. The length of the outer list is the number
    of shards required, and each inner list is the tuple of Tensors to use to
    determine the types and shapes of the corresponding shard. This method
    depends on the shard dimension, and calling it freezes the shard policy.

    Args:
      input_tensors: list of lists of Tensors. The outer list length corresponds
        to the desired number of shards, and each inner list is the size
        and shape of the desired configuration of the corresponding shard.

    Raises:
      ValueError: if any inner list is not a list of length
        self.number_of_tuple_elements; or the inner lists do not combine to
        form a consistent unsharded shape.
      TypeError: if the types of the Tensors in the inner lists do not match.
    """
    if not self._frozen:
        self._tuple_shapes = None
    number_of_shards = len(input_tensors)
    self.set_number_of_shards(number_of_shards)
    for t in input_tensors:
        if len(t) != self.number_of_tuple_elements:
            raise ValueError(f'input_tensors is {str(input_tensors)} but must be a list of lists, where each inner list has length number_of_tuple_elements={self.number_of_tuple_elements}')
    sharded_shapes = [[t[i].shape for t in input_tensors] for i in range(self.number_of_tuple_elements)]
    unsharded_shapes = [policy.get_unsharded_shape(s) for policy, s in zip(self._sharding_policies, sharded_shapes)]
    self.set_tuple_shapes(unsharded_shapes)
    for i in range(1, self.number_of_shards):
        for t1, t2 in zip(input_tensors[0], input_tensors[i]):
            if t1.dtype != t2.dtype:
                raise TypeError(f'types of the tuple elements of input_tensors {str(input_tensors)} are not consistent')
    self.set_tuple_types([t.dtype for t in input_tensors[0]])