from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_mnlogit_factor():
    dta = sm.datasets.anes96.load_pandas()
    dta['endog'] = dta.endog.replace(dict(zip(range(7), 'ABCDEFG')))
    exog = sm.add_constant(dta.exog, prepend=True)
    mod = sm.MNLogit(dta.endog, exog)
    res = mod.fit(disp=0)
    params = res.params
    summary = res.summary()
    predicted = res.predict(exog.iloc[:5, :])
    endogn = dta['endog']
    endogn.name = None
    mod = sm.MNLogit(endogn, exog)
    mod = smf.mnlogit('PID ~ ' + ' + '.join(dta.exog.columns), dta.data)
    res2 = mod.fit(disp=0)
    params_f = res2.params
    summary = res2.summary()
    assert_allclose(params_f, params, rtol=1e-10)
    predicted_f = res2.predict(dta.exog.iloc[:5, :])
    assert_allclose(predicted_f, predicted, rtol=1e-10)