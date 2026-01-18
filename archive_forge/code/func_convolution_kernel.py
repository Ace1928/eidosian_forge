import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def convolution_kernel(self, name='convolution_kernel'):
    """Convolution kernel corresponding to `self.spectrum`.

    The `D` dimensional DFT of this kernel is the frequency domain spectrum of
    this operator.

    Args:
      name:  A name to give this `Op`.

    Returns:
      `Tensor` with `dtype` `self.dtype`.
    """
    with self._name_scope(name):
        h = self._ifft(_to_complex(self.spectrum))
        return math_ops.cast(h, self.dtype)