import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
@pytest.mark.parametrize('a, b, x, result', [(1, 1, 0.44, 1.552707218511336), (-1, 1, 0.44, 0.56), (100, 100, 0.89, 2.4351296512898744), (-100, 100, 0.89, 0.407390624907681), (1.5, 100, 59.99, 3.8073513625965596), (-1.5, 100, 59.99, 0.25099240047125826)])
def test_geometric_convergence(self, a, b, x, result):
    assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)