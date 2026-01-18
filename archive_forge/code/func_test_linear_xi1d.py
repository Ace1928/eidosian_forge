import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_linear_xi1d(self):
    points, values = self._get_sample_4d_2()
    interp = RegularGridInterpolator(points, values)
    sample = np.asarray([0.1, 0.1, 10.0, 9.0])
    wanted = 1001.1
    assert_array_almost_equal(interp(sample), wanted)