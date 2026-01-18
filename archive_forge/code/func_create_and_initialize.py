import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def create_and_initialize(self):
    if callable(self._initial_value):
        initial_value = self._initial_value()
    with ops.device(initial_value.device):
        initial_value, shape, dtype, handle, handle_name, unique_id = _infer_shape_dtype_and_create_handle(initial_value, self._shape, self._dtype, self._name)
        self.initialize()
    super().__init__(trainable=self._trainable, shape=shape, dtype=dtype, handle=handle, synchronization=self._synchronization, constraint=self._constraint, aggregation=self._aggregation, distribute_strategy=self._distribute_strategy, name=self._name, unique_id=unique_id, handle_name=handle_name, graph_element=None, initial_value=initial_value, initializer_op=None, is_initialized_op=None, cached_value=None, caching_device=None)