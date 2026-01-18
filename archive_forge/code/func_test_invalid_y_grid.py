import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_invalid_y_grid():
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as w:
        etest_poisson_2indep(1, 1, 1, 1, ygrid=[1])
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert 'ygrid' in str(w[0].message)
    with pytest.raises(ValueError) as e:
        etest_poisson_2indep(1, 1, 1, 1, y_grid=1)
    assert 'y_grid' in str(e.value)