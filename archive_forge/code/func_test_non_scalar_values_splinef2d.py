import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_non_scalar_values_splinef2d(self):
    points, values = self._sample_4d_data()
    np.random.seed(1234)
    values = np.random.rand(3, 3, 3, 3, 6)
    sample = np.random.rand(7, 11, 4)
    assert_raises(ValueError, interpn, points, values, sample, method='splinef2d')