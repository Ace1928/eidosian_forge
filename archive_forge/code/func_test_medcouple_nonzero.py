import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_medcouple_nonzero(self):
    mc = medcouple(np.array([1, 2, 7, 9, 10.0]))
    assert_almost_equal(mc, -0.3333333)