import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_min_nobs(self):
    x = np.arange(3.0)
    with pytest.raises(ValueError):
        lilliefors(x, dist='norm', pvalmethod='approx')
    x = np.arange(2.0)
    with pytest.raises(ValueError):
        lilliefors(x, dist='exp', pvalmethod='approx')