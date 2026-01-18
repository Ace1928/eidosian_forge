from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
def get_var_on_device(self, device):
    for i, d in enumerate(self._devices):
        if d == device:
            return self._distributed_variables[i]
    raise ValueError('Device %s is not found' % device)