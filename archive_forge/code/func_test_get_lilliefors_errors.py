import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_get_lilliefors_errors(reset_randomstate):
    with pytest.raises(ValueError):
        get_lilliefors_table(dist='unknown')
    with pytest.raises(ValueError):
        kstest_fit(np.random.standard_normal(100), dist='unknown', pvalmethod='table')