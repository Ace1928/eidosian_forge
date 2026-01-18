import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def reduce_all(input_tensor, axis=None, keepdims=False):
    """A version of tf.reduce_all that eagerly evaluates if possible."""
    v = get_static_value(input_tensor)
    if v is None:
        return math_ops.reduce_all(input_tensor, axis=axis, keepdims=keepdims)
    else:
        return v.all(axis=axis, keepdims=keepdims)