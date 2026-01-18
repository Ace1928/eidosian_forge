import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('points_values, sample', [(_get_sample_4d, np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])), (_get_sample_4d_2, np.asarray([0.1, 0.1, 10.0, 9.0]))])
def test_linear_and_slinear_close(self, points_values, sample):
    points, values = points_values(self)
    interp = RegularGridInterpolator(points, values, method='linear')
    v1 = interp(sample)
    interp = RegularGridInterpolator(points, values, method='slinear')
    v2 = interp(sample)
    assert_allclose(v1, v2)