import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
def resource_handle_call_time_value(self):
    """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

    def closure():
        dispatch_context = coordinator_context.get_current_dispatch_context()
        if dispatch_context:
            local_resource_restore_context = get_current_local_resource_restore_context()
            if local_resource_restore_context:
                remote_value = local_resource_restore_context.instance.resource_handle
            else:
                remote_value = self._distributed_table._values[dispatch_context.worker_index]
            ret = dispatch_context.maybe_get_remote_value(remote_value)
            return ret
        else:
            return self._coordinator_instance.resource_handle
    return (closure, tensor.TensorSpec(shape=(), dtype=dtypes.resource))