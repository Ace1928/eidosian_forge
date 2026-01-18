from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
def get_var_on_current_device(self):
    current_device = device_util.canonicalize(device_util.current())
    return self.get_var_on_device(current_device)