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
def on_write_assign_sub(var, value, use_locking=False, name=None, read_value=True):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return var._update(update_fn=assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)