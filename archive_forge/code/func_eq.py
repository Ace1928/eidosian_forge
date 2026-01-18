from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def eq(a, b):
    """Functional form of "equal"."""
    if tensor_util.is_tf_type(a) or tensor_util.is_tf_type(b):
        return _tf_equal(a, b)
    return _py_equal(a, b)