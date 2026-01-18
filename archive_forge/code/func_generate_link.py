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
def generate_link(flag, np_fun_name):
    """Generates link from numpy function name.

  Args:
    flag: the flag to control link form. See `set_np_doc_form`.
    np_fun_name: the numpy function name.

  Returns:
    A string.
  """
    if flag == 'dev':
        template = 'https://numpy.org/devdocs/reference/generated/numpy.%s.html'
    elif flag == 'stable':
        template = 'https://numpy.org/doc/stable/reference/generated/numpy.%s.html'
    elif re.match('\\d+(\\.\\d+(\\.\\d+)?)?$', flag):
        template = f'https://numpy.org/doc/{flag}/reference/generated/numpy.%s.html'
    else:
        return None
    return template % np_fun_name