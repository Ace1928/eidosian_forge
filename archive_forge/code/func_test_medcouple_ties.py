import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_medcouple_ties(self, reset_randomstate):
    x = np.array([1, 2, 2, 3, 4])
    mc = medcouple(x)
    assert_almost_equal(mc, 1.0 / 6.0)