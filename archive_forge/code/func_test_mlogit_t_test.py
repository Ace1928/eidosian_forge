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
def test_mlogit_t_test():
    data = sm.datasets.anes96.load()
    exog = sm.add_constant(data.exog, prepend=False)
    res1 = sm.MNLogit(data.endog, exog).fit(disp=0)
    r = np.ones(res1.cov_params().shape[0])
    t1 = res1.t_test(r)
    f1 = res1.f_test(r)
    exog = sm.add_constant(data.exog, prepend=False)
    endog, exog = (np.asarray(data.endog), np.asarray(exog))
    res2 = sm.MNLogit(endog, exog).fit(disp=0)
    t2 = res2.t_test(r)
    f2 = res2.f_test(r)
    assert_allclose(t1.effect, t2.effect)
    assert_allclose(f1.statistic, f2.statistic)
    tt = res1.t_test(np.eye(np.size(res2.params)))
    assert_allclose(tt.tvalue.reshape(6, 6, order='F'), res1.tvalues.to_numpy())
    tt = res2.t_test(np.eye(np.size(res2.params)))
    assert_allclose(tt.tvalue.reshape(6, 6, order='F'), res2.tvalues)
    wt = res1.wald_test(np.eye(np.size(res2.params))[0], scalar=True)
    assert_allclose(wt.pvalue, res1.pvalues.to_numpy()[0, 0])
    tt = res1.t_test('y1_logpopul')
    wt = res1.wald_test('y1_logpopul', scalar=True)
    assert_allclose(tt.pvalue, wt.pvalue)
    wt = res1.wald_test('y1_logpopul, y2_logpopul', scalar=True)
    assert_allclose(wt.statistic, 5.68660562, rtol=1e-08)