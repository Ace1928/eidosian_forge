import numpy as np
from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform
def test_bs_0degree():
    x = np.logspace(-1, 1, 10)
    result = bs(x, knots=[1, 4], degree=0, include_intercept=True)
    assert result.shape[1] == 3
    expected_0 = np.zeros(10)
    expected_0[x < 1] = 1
    assert np.array_equal(result[:, 0], expected_0)
    expected_1 = np.zeros(10)
    expected_1[(x >= 1) & (x < 4)] = 1
    assert np.array_equal(result[:, 1], expected_1)
    expected_2 = np.zeros(10)
    expected_2[x >= 4] = 1
    assert np.array_equal(result[:, 2], expected_2)
    assert np.array_equal(bs([0, 1, 2], degree=0, knots=[1], include_intercept=True), [[1, 0], [0, 1], [0, 1]])
    result_int = bs(x, knots=[1, 4], degree=0, include_intercept=True)
    result_no_int = bs(x, knots=[1, 4], degree=0, include_intercept=False)
    assert np.array_equal(result_int[:, 1:], result_no_int)