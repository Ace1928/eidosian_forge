from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def cache_variable_reads():
    """Scope for caching variable reads for AggregatingVariable.

  The variable reads for AggregatingVariable inside this scope are cached. i.e.
  the first read of variable reads the value from possibly remote handle, but
  subsequent reads are returned using local cached value.

  For example:
  strategy = ParameterServerStrategy...
  with strategy.scope():
    # Variable v is of AggregatingVariable type with actual variable residing
    # on PS.
    v = tf.Variable(1.0)

  with distribute_utils.cache_variable_reads():
    v.read_value()  # Reads value 1.0
    v.assign(constant_op.constant(5.0))  # v changes to 5.0
    t1 = v.read_value()
    t2 = v.read_value()  # Both t1 & t2 return cached value 1.0 from local CPU.

  Notes about cache_variable_reads scope:
  1. Nesting of scope cache_variable_reads() is not supported
  2. And when caching scope is enabled, the thread enabling the cache and
    mirrored_run._MirroredReplicaThread threads spawned from it will have
    caching enabled.

  Yields:
    A context for caching variables.
  """
    try:
        if caching_scope_local.in_caching_scope():
            raise ValueError('cache_variable_reads scope cannot be nested')
        caching_scope_local.enter_scope()
        yield
    finally:
        caching_scope_local.exit_scope()