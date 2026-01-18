import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def default_float_type():
    """Gets the default float type.

  Returns:
    If `is_prefer_float32()` is false and `is_allow_float64()` is true, returns
    float64; otherwise returns float32.
  """
    if not is_prefer_float32() and is_allow_float64():
        return float64
    else:
        return float32