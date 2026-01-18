import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_durbin_watson_pandas(reset_randomstate):
    x = np.random.randn(50)
    x_series = pd.Series(x)
    assert_almost_equal(durbin_watson(x), durbin_watson(x_series), decimal=13)