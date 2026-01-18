from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, name=None):
    """Outputs random values from a normal distribution.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
        distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random normal values.
    """
    with ops.name_scope(name, 'stateful_normal', [shape, mean, stddev]) as name:
        shape = _shape_tensor(shape)
        mean = ops.convert_to_tensor(mean, dtype=dtype, name='mean')
        stddev = ops.convert_to_tensor(stddev, dtype=dtype, name='stddev')
        rnd = self._standard_normal(shape, dtype=dtype)
        return math_ops.add(rnd * stddev, mean, name=name)