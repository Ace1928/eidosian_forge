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
def set_configuration_from_input_tensors(self, input_tensors):
    """Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of Tensors whose types and shapes are used
    to set the queue configuration.

    Args:
      input_tensors: list of Tensors of the same types and shapes as
        the desired queue Tuple.

    Raises:
      ValueError: if input_tensors is not a list of length
        self.number_of_tuple_elements
    """
    if len(input_tensors) != self.number_of_tuple_elements:
        raise ValueError(f'input_tensors is {str(input_tensors)}, but should be a list of {self.number_of_tuple_elements} Tensors')
    self.set_tuple_shapes([t.shape for t in input_tensors])
    self.set_tuple_types([t.dtype for t in input_tensors])