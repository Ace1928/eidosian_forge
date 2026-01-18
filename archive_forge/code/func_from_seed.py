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
def from_seed(cls, seed, alg=None):
    """Creates a generator from a seed.

    A seed is a 1024-bit unsigned integer represented either as a Python
    integer or a vector of integers. Seeds shorter than 1024-bit will be
    padded. The padding, the internal structure of a seed and the way a seed
    is converted to a state are all opaque (unspecified). The only semantics
    specification of seeds is that two different seeds are likely to produce
    two independent generators (but no guarantee).

    Args:
      seed: the seed for the RNG.
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if alg is None:
        alg = DEFAULT_ALGORITHM
    alg = random_ops_util.convert_alg_to_int(alg)
    state = create_rng_state(seed, alg)
    return cls(state=state, alg=alg)