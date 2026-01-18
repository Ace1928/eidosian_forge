import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_complex_spline2fd(self):
    x, y, values = self._sample_2d_data()
    points = (x, y)
    values = values - 2j * values
    sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    with assert_warns(ComplexWarning):
        interpn(points, values, sample, method='splinef2d')