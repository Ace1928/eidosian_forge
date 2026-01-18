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
def test_loglike_vs_R(setup_model):
    model, params, results_R = setup_model
    loglike = model.loglike(params)
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R['loglik'][0] + const
    assert_allclose(loglike, loglike_R, rtol=1e-05, atol=1e-05)