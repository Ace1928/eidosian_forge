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
@classmethod
def from_key_counter(cls, key, counter, alg):
    """Creates a generator from a key and a counter.

    This constructor only applies if the algorithm is a counter-based algorithm.
    See method `key` for the meaning of "key" and "counter".

    Args:
      key: the key for the RNG, a scalar of type STATE_TYPE.
      counter: a vector of dtype STATE_TYPE representing the initial counter for
        the RNG, whose length is algorithm-specific.,
      alg: the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    alg = random_ops_util.convert_alg_to_int(alg)
    counter.shape.assert_is_compatible_with([_get_state_size(alg) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    return cls(state=state, alg=alg)