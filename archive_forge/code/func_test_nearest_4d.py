import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_nearest_4d(self):
    points, values = self._sample_4d_data()
    interp_rg = RegularGridInterpolator(points, values, method='nearest')
    sample = np.asarray([[0.1, 0.1, 10.0, 9.0]])
    wanted = interpn(points, values, sample, method='nearest')
    assert_array_almost_equal(interp_rg(sample), wanted)