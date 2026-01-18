import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_influence_dtype():
    y = np.ones(20)
    np.random.seed(123)
    x = np.random.randn(20, 3)
    res1 = OLS(y, x).fit()
    res2 = OLS(y * 1.0, x).fit()
    cr1 = res1.get_influence().cov_ratio
    cr2 = res2.get_influence().cov_ratio
    assert_allclose(cr1, cr2, rtol=1e-14)
    cr3 = np.array([1.22239215, 1.31551021, 1.52671069, 1.05003921, 0.89099323, 1.57405066, 1.03230092, 0.95844196, 1.15531836, 1.21963623, 0.87699564, 1.16707748, 1.10481391, 0.98839447, 1.08999334, 1.35680102, 1.46227715, 1.45966708, 1.13659521, 1.22799038])
    assert_almost_equal(cr1, cr3, decimal=8)