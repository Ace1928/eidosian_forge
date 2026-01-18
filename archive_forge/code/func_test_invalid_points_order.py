import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_invalid_points_order(self):
    x = np.array([0.5, 2.0, 0.0, 4.0, 5.5])
    y = np.array([0.5, 2.0, 3.0, 4.0, 5.5])
    z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
    match = 'must be strictly ascending or descending'
    with pytest.raises(ValueError, match=match):
        interpn((x, y), z, xi)