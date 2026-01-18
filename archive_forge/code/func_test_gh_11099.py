import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
@pytest.mark.parametrize('a, b, x, desired', [(-1, -2, 2, 2), (-1, -4, 10, 3.5), (-2, -2, 1, 2.5)])
def test_gh_11099(self, a, b, x, desired):
    assert sc.hyp1f1(a, b, x) == desired