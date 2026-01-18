import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_ksstat(reset_randomstate):
    x = np.random.uniform(0, 1, 100)
    two_sided = ksstat(x, 'uniform', alternative='two_sided')
    greater = ksstat(x, 'uniform', alternative='greater')
    lower = ksstat(x, stats.uniform, alternative='lower')
    print(two_sided, greater, lower)
    assert lower <= two_sided
    assert greater <= two_sided