import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Probit
from statsmodels.treatment.treatment_effects import (
from .results import results_teffects as res_st
@pytest.mark.parametrize('case', methods)
def test_effects(self, case):
    meth, res2 = case
    teff = self.teff
    res1 = getattr(teff, meth)(return_results=False)
    assert_allclose(res1[:2], res2.table[:2, 0], rtol=0.0001)
    res0 = getattr(teff, meth)(return_results=True)
    assert_allclose(res1, res0.effect, rtol=0.0001)
    res1 = res0.results_gmm
    assert_allclose(res0.start_params, res1.params, rtol=1e-05)
    assert_allclose(res1.params[:2], res2.table[:2, 0], rtol=1e-05)
    assert_allclose(res1.bse[:2], res2.table[:2, 1], rtol=0.001)
    assert_allclose(res1.tvalues[:2], res2.table[:2, 2], rtol=0.001)
    assert_allclose(res1.pvalues[:2], res2.table[:2, 3], rtol=0.0001, atol=1e-15)
    ci = res1.conf_int()
    assert_allclose(ci[:2, 0], res2.table[:2, 4], rtol=0.0005)
    assert_allclose(ci[:2, 1], res2.table[:2, 5], rtol=0.0005)
    k_p = len(res1.params)
    if k_p == 8:
        idx = [0, 1, 7, 2, 3, 4, 5, 6]
    elif k_p == 18:
        idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10, 17, 12, 13, 14, 15, 16]
    elif k_p == 12:
        idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10]
    else:
        idx = np.arange(k_p)
    assert_allclose(res1.params, res2.table[idx, 0], rtol=0.0001)
    assert_allclose(res1.bse, res2.table[idx, 1], rtol=0.05)
    if not meth.startswith('aipw'):
        table = res2.table_t
        res1 = getattr(teff, meth)(return_results=False, effect_group=1)
        assert_allclose(res1[:2], table[:2, 0], rtol=0.0001)
        res0 = getattr(teff, meth)(return_results=True, effect_group=1)
        assert_allclose(res1, res0.effect, rtol=2e-05)
        res1 = res0.results_gmm
        assert_allclose(res0.start_params, res1.params, rtol=5e-05)
        assert_allclose(res1.params[:2], table[:2, 0], rtol=5e-05)
        assert_allclose(res1.bse[:2], table[:2, 1], rtol=0.001)
        assert_allclose(res1.tvalues[:2], table[:2, 2], rtol=0.001)
        assert_allclose(res1.pvalues[:2], table[:2, 3], rtol=0.0001, atol=1e-15)
        ci = res1.conf_int()
        assert_allclose(ci[:2, 0], table[:2, 4], rtol=0.0005)
        assert_allclose(ci[:2, 1], table[:2, 5], rtol=0.0005)
        res1 = getattr(teff, meth)(return_results=False, effect_group=0)
        res0 = getattr(teff, meth)(return_results=True, effect_group=0)
        assert_allclose(res1, res0.effect, rtol=1e-12)
        assert_allclose(res0.start_params, res0.results_gmm.params, rtol=1e-12)