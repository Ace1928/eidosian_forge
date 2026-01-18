import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_medcouple_symmetry(self, reset_randomstate):
    x = np.random.standard_normal(100)
    mcp = medcouple(x)
    mcn = medcouple(-x)
    assert_almost_equal(mcp + mcn, 0)