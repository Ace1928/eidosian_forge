import numpy as np
from numpy.testing import assert_
from scipy.signal import (decimate,
def test_decimate():
    ones_f32 = np.ones(32, dtype=np.float32)
    assert_(decimate(ones_f32, 2).dtype == np.float32)
    ones_i64 = np.ones(32, dtype=np.int64)
    assert_(decimate(ones_i64, 2).dtype == np.float64)