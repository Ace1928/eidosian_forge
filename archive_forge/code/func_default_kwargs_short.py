from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
@pytest.fixture(scope='function')
def default_kwargs_short():
    kwargs = default_kwargs_base()
    y = kwargs['y'][:-1]
    nobs = y.shape[0]
    work = np.zeros((nobs + 2 * kwargs['np'], 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    kwargs.update(dict(y=y, n=nobs, rw=rw, trend=trend, season=season, work=work))
    return kwargs