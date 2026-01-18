import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_effectsize_fstat():
    Eta_Sq_partial = 0.796983758700696
    CI_eta2 = (0.685670133284926, 0.855981325777856)
    Epsilon_Sq_partial = 0.779582366589327
    CI_eps2 = (0.658727573280777, 0.843636867987386)
    Omega_Sq_partial = 0.775086505190311
    CI_omega2 = (0.65286429480169, 0.840179680453464)
    Cohens_f_partial = 1.98134153686695
    CI_f = (1.47694659580859, 2.43793847155554)
    f_stat, df1, df2 = (45.8, 3, 35)
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(np.sqrt(fes.f2), Cohens_f_partial, rtol=1e-13)
    assert_allclose(fes.eta2, Eta_Sq_partial, rtol=1e-13)
    assert_allclose(fes.eps2, Epsilon_Sq_partial, rtol=1e-13)
    assert_allclose(fes.omega2, Omega_Sq_partial, rtol=1e-13)
    ci_nc = confint_noncentrality(f_stat, (df1, df2), alpha=0.1)
    ci_es = smo._fstat2effectsize(ci_nc / df1, (df1, df2))
    assert_allclose(ci_es.eta2, CI_eta2, rtol=0.0002)
    assert_allclose(ci_es.eps2, CI_eps2, rtol=0.0002)
    assert_allclose(ci_es.omega2, CI_omega2, rtol=0.0002)
    assert_allclose(np.sqrt(ci_es.f2), CI_f, rtol=0.0002)