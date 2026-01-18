import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test__get_all_sorted_knots():
    import pytest
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), -1)
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), 0)
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), 0, lower_bound=1)
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), 0, upper_bound=5)
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), 0, lower_bound=3, upper_bound=1)
    assert np.array_equal(_get_all_sorted_knots(np.array([]), 0, lower_bound=1, upper_bound=5), [1, 5])
    pytest.raises(ValueError, _get_all_sorted_knots, np.array([]), 0, lower_bound=1, upper_bound=1)
    x = np.arange(6) * 2
    pytest.raises(ValueError, _get_all_sorted_knots, x, -2)
    assert np.array_equal(_get_all_sorted_knots(x, 0), [0, 10])
    assert np.array_equal(_get_all_sorted_knots(x, 0, lower_bound=3, upper_bound=8), [3, 8])
    assert np.array_equal(_get_all_sorted_knots(x, 2, lower_bound=1, upper_bound=9), [1, 4, 6, 9])
    pytest.raises(ValueError, _get_all_sorted_knots, x, 2, lower_bound=1, upper_bound=3)
    pytest.raises(ValueError, _get_all_sorted_knots, x, 1, lower_bound=1.3, upper_bound=1.4)
    assert np.array_equal(_get_all_sorted_knots(x, 1, lower_bound=1, upper_bound=3), [1, 2, 3])
    pytest.raises(ValueError, _get_all_sorted_knots, x, 1, lower_bound=2, upper_bound=3)
    pytest.raises(ValueError, _get_all_sorted_knots, x, 1, inner_knots=[2, 3])
    pytest.raises(ValueError, _get_all_sorted_knots, x, lower_bound=2, upper_bound=3)
    assert np.array_equal(_get_all_sorted_knots(x, inner_knots=[3, 7]), [0, 3, 7, 10])
    assert np.array_equal(_get_all_sorted_knots(x, inner_knots=[3, 7], lower_bound=2), [2, 3, 7, 10])
    pytest.raises(ValueError, _get_all_sorted_knots, x, inner_knots=[3, 7], lower_bound=4)
    pytest.raises(ValueError, _get_all_sorted_knots, x, inner_knots=[3, 7], upper_bound=6)