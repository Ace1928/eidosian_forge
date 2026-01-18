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
def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    return self._apply_update(self._variable.scatter_sub, sparse_delta, use_locking, name)