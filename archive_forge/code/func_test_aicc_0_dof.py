from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_aicc_0_dof():
    endog = [109.0, 101.0, 104.0, 90.0, 105.0]
    model = ETSModel(endog=endog, initialization_method='known', initial_level=100.0, initial_trend=0.0, error='add', trend='add', damped_trend=True)
    aicc = model.fit().aicc
    assert not np.isfinite(aicc)
    assert aicc > 0