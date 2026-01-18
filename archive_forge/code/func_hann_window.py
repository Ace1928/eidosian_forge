import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.hann_window')
@dispatch.add_dispatch_support
def hann_window(window_length, periodic=True, dtype=dtypes.float32, name=None):
    """Generate a [Hann window][hann].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window. Periodic windows are typically used for spectral
      analysis while symmetric windows are typically used for digital
      filter design.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type.

  [hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
  """
    return _raised_cosine_window(name, 'hann_window', window_length, periodic, dtype, 0.5, 0.5)