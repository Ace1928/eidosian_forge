import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_fillvalue_type(self):
    values = np.ones((10, 20, 30), dtype='>f4')
    points = [np.arange(n) for n in values.shape]
    RegularGridInterpolator(points, values)
    RegularGridInterpolator(points, values, fill_value=0.0)