import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_durbin_watson_3d(self, reset_randomstate):
    shape = (10, 1, 10)
    x = np.random.standard_normal(100)
    dw = sum(np.diff(x) ** 2.0) / np.dot(x, x)
    x = np.tile(x[None, :, None], shape)
    assert_almost_equal(np.squeeze(dw * np.ones(shape)), durbin_watson(x, axis=1))