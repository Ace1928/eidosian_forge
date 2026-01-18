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
def exponential_power_convolution_kernel(grid_shape, length_scale, power=None, divisor=None, zero_inflation=None):
    """Make an exponentiated convolution kernel.

  In signal processing, a [kernel]
  (https://en.wikipedia.org/wiki/Kernel_(image_processing)) `h` can be convolved
  with a signal `x` to filter its spectral content.

  This function makes a `d-dimensional` convolution kernel `h` of shape
  `grid_shape = [N0, N1, ...]`. For `n` a multi-index with `n[i] < Ni / 2`,

  ```h[n] = exp{sum(|n / (length_scale * grid_shape)|**power) / divisor}.```

  For other `n`, `h` is extended to be circularly symmetric. That is

  ```h[n0 % N0, ...] = h[(-n0) % N0, ...]```

  Since `h` is circularly symmetric and real valued, `H = FFTd[h]` is the
  spectrum of a symmetric (real) circulant operator `A`.

  #### Example uses

  ```
  # Matern one-half kernel, d=1.
  # Will be positive definite without zero_inflation.
  h = exponential_power_convolution_kernel(
      grid_shape=[10], length_scale=[0.1], power=1)
  A = LinearOperatorCirculant(
      tf.signal.fft(tf.cast(h, tf.complex64)),
      is_self_adjoint=True, is_positive_definite=True)

  # Gaussian RBF kernel, d=3.
  # Needs zero_inflation since `length_scale` is long enough to cause aliasing.
  h = exponential_power_convolution_kernel(
      grid_shape=[10, 10, 10], length_scale=[0.1, 0.2, 0.2], power=2,
      zero_inflation=0.15)
  A = LinearOperatorCirculant3D(
      tf.signal.fft3d(tf.cast(h, tf.complex64)),
      is_self_adjoint=True, is_positive_definite=True)
  ```

  Args:
    grid_shape: Length `d` (`d` in {1, 2, 3}) list-like of Python integers. The
      shape of the grid on which the convolution kernel is defined.
    length_scale: Length `d` `float` `Tensor`. The scale at which the kernel
      decays in each direction, as a fraction of `grid_shape`.
    power: Scalar `Tensor` of same `dtype` as `length_scale`, default `2`.
      Higher (lower) `power` results in nearby points being more (less)
      correlated, and far away points being less (more) correlated.
    divisor: Scalar `Tensor` of same `dtype` as `length_scale`. The slope of
      decay of `log(kernel)` in terms of fractional grid points, along each
      axis, at `length_scale`, is `power/divisor`. By default, `divisor` is set
      to `power`. This means, by default, `power=2` results in an exponentiated
      quadratic (Gaussian) kernel, and `power=1` is a Matern one-half.
    zero_inflation: Scalar `Tensor` of same `dtype` as `length_scale`, in
      `[0, 1]`. Let `delta` be the Kronecker delta. That is,
      `delta[0, ..., 0] = 1` and all other entries are `0`. Then
      `zero_inflation` modifies the return value via
      `h --> (1 - zero_inflation) * h + zero_inflation * delta`. This may be
      needed to ensure a positive definite kernel, especially if `length_scale`
      is large enough for aliasing and `power > 1`.

  Returns:
    `Tensor` of shape `grid_shape` with same `dtype` as `length_scale`.
  """
    nd = len(grid_shape)
    length_scale = tensor_conversion.convert_to_tensor_v2_with_dispatch(length_scale, name='length_scale')
    dtype = length_scale.dtype
    power = 2.0 if power is None else power
    power = tensor_conversion.convert_to_tensor_v2_with_dispatch(power, name='power', dtype=dtype)
    divisor = power if divisor is None else divisor
    divisor = tensor_conversion.convert_to_tensor_v2_with_dispatch(divisor, name='divisor', dtype=dtype)
    zero = math_ops.cast(0.0, dtype)
    one = math_ops.cast(1.0, dtype)
    ts = [math_ops.linspace(zero, one, num=n) for n in grid_shape]
    log_vals = []
    for i, x in enumerate(array_ops.meshgrid(*ts, indexing='ij')):
        midpoint = ts[i][math_ops.cast(math_ops.floor(one / 2.0 * grid_shape[i]), dtypes.int32)]
        log_vals.append(-math_ops.abs((x - midpoint) / length_scale[i]) ** power / divisor)
    kernel = math_ops.exp(fft_ops.ifftshift(sum(log_vals), axes=[-i for i in range(1, nd + 1)]))
    if zero_inflation:
        zero_inflation = tensor_conversion.convert_to_tensor_v2_with_dispatch(zero_inflation, name='zero_inflation', dtype=dtype)
        delta = array_ops.pad(array_ops.reshape(one, [1] * nd), [[0, dim - 1] for dim in grid_shape])
        kernel = (1.0 - zero_inflation) * kernel + zero_inflation * delta
    return kernel