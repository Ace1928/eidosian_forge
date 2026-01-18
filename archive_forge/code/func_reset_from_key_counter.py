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
def reset_from_key_counter(self, key, counter):
    """Resets the generator by a new key-counter pair.

    See `from_key_counter` for the meaning of "key" and "counter".

    Args:
      key: the new key.
      counter: the new counter.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    counter.shape.assert_is_compatible_with([_get_state_size(self.algorithm) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    self._state_var.assign(state)