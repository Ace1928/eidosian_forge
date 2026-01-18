import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
def initialize_all(self):
    """Initialize all uninitialized lazy variables stored in scope."""

    def assign_function(uninitialized_var_list):
        for var in uninitialized_var_list:
            val = var._initial_value
            packed_var = getattr(var, '_packed_var', None)
            handle = getattr(packed_var, 'packed_handle', var.handle)
            with ops.device(handle.device):
                resource_variable_ops.AssignVariableOp(resource=handle, value=val)
        return constant_op.constant([])
    assign_tf_function = def_function.function(assign_function, autograph=False, jit_compile=False)
    with ops.init_scope():
        if len(self._uninitialized_var_list) > 1:
            assign_tf_function(self._uninitialized_var_list)
        else:
            assign_function(self._uninitialized_var_list)
    self._uninitialized_var_list = []