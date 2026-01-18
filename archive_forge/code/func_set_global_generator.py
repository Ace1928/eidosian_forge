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
@tf_export('random.set_global_generator', 'random.experimental.set_global_generator')
def set_global_generator(generator):
    """Replaces the global generator with another `Generator` object.

  This function replaces the global generator with the provided `generator`
  object.
  A random number generator utilizes a `tf.Variable` object to store its state.
  The user shall be aware of caveats how `set_global_generator` interacts with
  `tf.function`:

  - tf.function puts restrictions on Variable creation thus one cannot freely
    create a new random generator instance inside `tf.function`.
    To call `set_global_generator` inside `tf.function`, the generator instance
    must have already been created eagerly.
  - tf.function captures the Variable during trace-compilation, thus a compiled
    f.function will not be affected `set_global_generator` as demonstrated by
    random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .

  For most use cases, avoid calling `set_global_generator` after program
  initialization, and prefer to reset the state of the existing global generator
  instead, such as,

  >>> rng = tf.random.get_global_generator()
  >>> rng.reset_from_seed(30)


  Args:
    generator: the new `Generator` object.
  """
    global global_generator
    global_generator = generator