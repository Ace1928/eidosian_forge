import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_epsilon_not_specified_error(self):
    y = np.linspace(0, 1, 5)[:, None]
    d = np.zeros(5)
    for kernel in _AVAILABLE:
        if kernel in _SCALE_INVARIANT:
            continue
        match = '`epsilon` must be specified'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, kernel=kernel)