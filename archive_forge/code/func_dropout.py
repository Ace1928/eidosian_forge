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
def dropout(i, do_dropout, v, n):
    if not isinstance(do_dropout, bool) or do_dropout:
        return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
    else:
        return v