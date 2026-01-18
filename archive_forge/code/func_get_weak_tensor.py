import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
def get_weak_tensor(*args, **kwargs):
    return WeakTensor.from_tensor(constant_op.constant(*args, **kwargs))