import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
def test_limit_check(self):
    result = sc.gammaincc(1e-10, 1)
    limit = sc.gammaincc(0, 1)
    assert np.isclose(result, limit)