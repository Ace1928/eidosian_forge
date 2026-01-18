from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_fit_misc():
    true = results_sarimax.wpi1_stationary
    endog = np.diff(true['data'])[1:]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 1), trend='c')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res1 = mod.fit(method='ncg', disp=0, optim_hessian='opg', optim_complex_step=False)
        res2 = mod.fit(method='ncg', disp=0, optim_hessian='oim', optim_complex_step=False)
    assert_allclose(res1.llf, res2.llf, rtol=0.01)
    mod, _ = get_dummy_mod(fit=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_params = mod.fit(disp=-1, return_params=True)
    assert_almost_equal(res_params, [0, 0], 5)