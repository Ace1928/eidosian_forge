from typing import List, Optional
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
def maybe_get_device_name(device_name):
    if device_name is None:
        device_name = random_ops.random_normal([]).device
    return device_name