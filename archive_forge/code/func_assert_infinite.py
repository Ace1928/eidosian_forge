from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def assert_infinite(val):
    val = make_val(val)
    if dtype in DTYPES_WITH_NO_INFINITY:
        self.assertTrue(np.isnan(val), f'expected NaN, got {val}')
    else:
        self.assertTrue(np.isposinf(val), f'expected inf, got {val}')