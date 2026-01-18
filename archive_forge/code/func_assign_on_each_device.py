from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object
def assign_on_each_device(var, assign_func, value, read_value):
    """Update the variable on each replica with the given assign_func and value."""
    if var._packed_variable is not None:
        update = control_flow_ops.group(tuple((assign_func(d, var._packed_variable, value) for d in var._devices)))
    else:
        update = control_flow_ops.group(tuple((assign_func(v.device, v, value) for v in var._values)))
    if not read_value:
        return update
    with ops.control_dependencies([update] if update else []):
        return var.read_value()