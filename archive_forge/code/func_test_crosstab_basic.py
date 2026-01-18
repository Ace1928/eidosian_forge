import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
@pytest.mark.parametrize('sparse', [False, True])
def test_crosstab_basic(sparse):
    a = [0, 0, 9, 9, 0, 0, 9]
    b = [2, 1, 3, 1, 2, 3, 3]
    expected_avals = [0, 9]
    expected_bvals = [1, 2, 3]
    expected_count = np.array([[1, 2, 1], [1, 0, 2]])
    (avals, bvals), count = crosstab(a, b, sparse=sparse)
    assert_array_equal(avals, expected_avals)
    assert_array_equal(bvals, expected_bvals)
    if sparse:
        assert_array_equal(count.A, expected_count)
    else:
        assert_array_equal(count, expected_count)