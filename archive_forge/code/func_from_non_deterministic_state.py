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
def from_non_deterministic_state(cls, alg=None):
    """Creates a generator by non-deterministically initializing its state.

    The source of the non-determinism will be platform- and time-dependent.

    Args:
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if config.is_op_determinism_enabled():
        raise RuntimeError('"from_non_deterministic_state" cannot be called when determinism is enabled.')
    if alg is None:
        alg = DEFAULT_ALGORITHM
    alg = random_ops_util.convert_alg_to_int(alg)
    state = non_deterministic_ints(shape=[_get_state_size(alg)], dtype=SEED_TYPE)
    return cls(state=state, alg=alg)