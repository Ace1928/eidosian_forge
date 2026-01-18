from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
def on_device(self, device):
    return PackedVarAndDevice(self, device)