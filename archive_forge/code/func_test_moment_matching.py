import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def test_moment_matching(samples, number_moments, dist, stride=0):
    """Return z-test scores for sample moments to match analytic moments.

  Given `samples`, check that the first sample `number_moments` match
  the given  `dist` moments by doing a z-test.

  Args:
    samples: Samples from target distribution.
    number_moments: Python `int` describing how many sample moments to check.
    dist: SciPy distribution object that provides analytic moments.
    stride: Distance between samples to check for statistical properties.
      A stride of 0 means to use all samples, while other strides test for
      spatial correlation.
  Returns:
    Array of z_test scores.
  """
    sample_moments = []
    expected_moments = []
    variance_sample_moments = []
    for i in range(1, number_moments + 1):
        if len(samples.shape) == 2:
            strided_range = samples.flat[::(i - 1) * stride + 1]
        else:
            strided_range = samples[::(i - 1) * stride + 1, ...]
        sample_moments.append(np.mean(strided_range ** i, axis=0))
        expected_moments.append(dist.moment(i))
        variance_sample_moments.append((dist.moment(2 * i) - dist.moment(i) ** 2) / len(strided_range))
    z_test_scores = []
    for i in range(1, number_moments + 1):
        total_variance = variance_sample_moments[i - 1] + i * np.finfo(samples.dtype).eps
        tiny = np.finfo(samples.dtype).tiny
        assert np.all(total_variance > 0)
        total_variance = np.where(total_variance < tiny, tiny, total_variance)
        z_test_scores.append(abs((sample_moments[i - 1] - expected_moments[i - 1]) / np.sqrt(total_variance)))
    return z_test_scores