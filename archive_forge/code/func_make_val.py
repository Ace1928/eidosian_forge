from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def make_val(val):
    return np.array(val, dtype=dtype)