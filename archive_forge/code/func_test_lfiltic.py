import numpy as np
from numpy.testing import assert_
from scipy.signal import (decimate,
def test_lfiltic():
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    x_f32 = np.ones(32, dtype=np.float32)
    b_f64 = b_f32.astype(np.float64)
    a_f64 = a_f32.astype(np.float64)
    x_f64 = x_f32.astype(np.float64)
    assert_(lfiltic(b_f64, a_f32, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f64, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f64).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f32, x_f64).dtype == np.float64)