import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_invalid_kernel_name_error(self):
    y = np.linspace(0, 1, 5)[:, None]
    d = np.zeros(5)
    match = '`kernel` must be one of'
    with pytest.raises(ValueError, match=match):
        self.build(y, d, kernel='test')