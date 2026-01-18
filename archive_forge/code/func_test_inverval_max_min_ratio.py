import numpy as np
import pytest
from sklearn.utils._plotting import _interval_max_min_ratio, _validate_score_name
@pytest.mark.parametrize('data, lower_bound, upper_bound', [(np.geomspace(0.1, 1, 5), 5, 6), (-np.geomspace(0.1, 1, 10), 7, 8), (np.linspace(0, 1, 5), 0.9, 1.1), ([1, 2, 5, 10, 20, 50], 20, 40)])
def test_inverval_max_min_ratio(data, lower_bound, upper_bound):
    assert lower_bound < _interval_max_min_ratio(data) < upper_bound