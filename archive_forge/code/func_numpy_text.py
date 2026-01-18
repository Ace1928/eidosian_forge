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
def numpy_text(tensor, is_repr=False):
    """Human readable representation of a tensor's numpy value."""
    if tensor.dtype.is_numpy_compatible:
        text = repr(tensor._numpy()) if is_repr else str(tensor._numpy())
    else:
        text = '<unprintable>'
    if '\n' in text:
        text = '\n' + text
    return text