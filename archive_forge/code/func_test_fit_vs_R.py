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
def test_fit_vs_R(setup_model, reset_randomstate):
    model, params, results_R = setup_model
    if PLATFORM_WIN and model.short_name == 'AAdA':
        start = params
    else:
        start = None
    fit = model.fit(disp=True, pgtol=1e-08, start_params=start)
    const = -model.nobs / 2 * (np.log(2 * np.pi / model.nobs) + 1)
    loglike_R = results_R['loglik'][0] + const
    loglike = fit.llf
    try:
        assert loglike >= loglike_R - 0.0001
    except AssertionError:
        fit = model.fit(disp=True, pgtol=1e-08, start_params=params)
        loglike = fit.llf
        try:
            assert loglike >= loglike_R - 0.0001
        except AssertionError:
            if PLATFORM_LINUX32:
                pytest.xfail('Known to fail on 32-bit Linux')
            else:
                raise