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
def get_on_write_saveable(var, primary_var, name):
    """Return saveable spec for AUTO and ON_WRITE variables."""

    def tensor():
        if context.executing_eagerly() and (not primary_var.is_initialized()):
            return None
        strategy = var.distribute_strategy
        return strategy.extended.read_var(var)
    spec = saveable_object.SaveSpec(tensor=tensor, slice_spec='', name=name, dtype=var.dtype, device=primary_var.device)
    return (tensor, [spec])