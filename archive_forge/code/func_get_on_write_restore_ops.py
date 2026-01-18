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
def get_on_write_restore_ops(var, tensor):
    """Return restore ops for AUTO and ON_WRITE variables."""
    packed_var = var._packed_variable
    if packed_var is not None:
        return control_flow_ops.group(tuple((assign_on_device(d, packed_var, tensor) for d in packed_var.devices)))
    return control_flow_ops.group(tuple((assign_on_device(v.device, v, tensor) for v in var.values)))