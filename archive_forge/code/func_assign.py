import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
def assign(self, value, use_locking=None, name=None, read_value=True):
    return self._apply_assign_update(self._variable.assign, value, use_locking, name, read_value)