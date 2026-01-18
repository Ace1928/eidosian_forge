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
def np_doc_only(np_fun_name, np_fun=None):
    """Attachs numpy docstring to a function.

  This differs from np_doc in that it doesn't check for a match in signature.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
    np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)

    def decorator(f):
        f.__doc__ = _np_doc_helper(f, np_fun, np_fun_name=np_fun_name)
        return f
    return decorator