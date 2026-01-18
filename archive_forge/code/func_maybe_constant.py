import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import util_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def maybe_constant(val):
    val_static = tensor_util.constant_value(val)
    return (val_static, True) if val_static is not None else (val, False)