import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def flat_wavelet(l, w):
    return np.full(w, 1 / w)