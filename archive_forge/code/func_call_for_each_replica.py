import functools
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def call_for_each_replica(self, fn, args=(), kwargs=None):
    """Run `fn` once per replica.

    This is a method that expected by the strategy base class in its `run()`.

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    """
    distribute_lib._require_cross_replica_or_default_context_extended(self)
    if kwargs is None:
        kwargs = {}
    map_fn = functools.partial(dtensor_util.convert_inputs_to_dtensor, mesh=self._mesh)
    d_args = nest.map_structure(map_fn, args)
    d_kwargs = nest.map_structure(map_fn, kwargs)
    with self._container_strategy().scope():
        with dtensor_util.DTensorReplicaContext(self._container_strategy()):
            dtensor_result = fn(*d_args, **d_kwargs)
    return nest.map_structure(dtensor_util.DTensorDistributedValue, dtensor_result)