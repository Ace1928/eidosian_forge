import numpy as np
from numpy.testing import assert_
from scipy.signal import (decimate,
def test_lfilter_zi():
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    assert_(lfilter_zi(b_f32, a_f32).dtype == np.float32)