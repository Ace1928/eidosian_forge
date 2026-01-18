import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_degree_warning(self):
    y = np.linspace(0, 1, 5)[:, None]
    d = np.zeros(5)
    for kernel, deg in _NAME_TO_MIN_DEGREE.items():
        match = f'`degree` should not be below {deg}'
        with pytest.warns(Warning, match=match):
            self.build(y, d, epsilon=1.0, kernel=kernel, degree=deg - 1)