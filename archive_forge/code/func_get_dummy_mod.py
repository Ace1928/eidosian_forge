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
def get_dummy_mod(fit=True, pandas=False):
    endog = np.arange(100) * 1.0
    exog = 2 * endog
    if pandas:
        index = pd.date_range('1960-01-01', periods=100, freq='MS')
        endog = pd.Series(endog, index=index)
        exog = pd.Series(exog, index=index)
    mod = sarimax.SARIMAX(endog, exog=exog, order=(0, 0, 0), time_varying_regression=True, mle_regression=False, use_exact_diffuse=True)
    if fit:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = mod.fit(disp=-1)
    else:
        res = None
    return (mod, res)