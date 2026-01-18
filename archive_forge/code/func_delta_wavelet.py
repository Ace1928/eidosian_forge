import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def delta_wavelet(s, t):
    return np.array([1])