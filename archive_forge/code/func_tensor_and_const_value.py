import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
def tensor_and_const_value(v):
    tensor_value = tensor_conversion.convert_to_tensor_v2_with_dispatch(v)
    const_value = tensor_util.constant_value(tensor_value)
    return (tensor_value, const_value)