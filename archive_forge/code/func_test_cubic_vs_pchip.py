import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_cubic_vs_pchip(self):
    x, y = ([1, 2, 3, 4], [1, 2, 3, 4])
    xg, yg = np.meshgrid(x, y, indexing='ij')
    values = (lambda x, y: x ** 4 * y ** 4)(xg, yg)
    cubic = RegularGridInterpolator((x, y), values, method='cubic')
    pchip = RegularGridInterpolator((x, y), values, method='pchip')
    vals_cubic = cubic([1.5, 2])
    vals_pchip = pchip([1.5, 2])
    assert not np.allclose(vals_cubic, vals_pchip, atol=1e-14, rtol=0)