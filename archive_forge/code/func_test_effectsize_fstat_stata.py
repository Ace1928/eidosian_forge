import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_effectsize_fstat_stata():
    eta2 = 0.2720398648288652
    lb_eta2 = 0.0742092468714613
    ub_eta2 = 0.4156116886974804
    omega2 = 0.2356418580703085
    lb_omega2 = 0.0279197092150344
    ub_omega2 = 0.3863922731323545
    f_stat, df1, df2 = (7.47403193349075, 2, 40)
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(fes.eta2, eta2, rtol=1e-13)
    assert_allclose(fes.omega2, omega2, rtol=0.02)
    ci_es = smo.confint_effectsize_oneway(f_stat, (df1, df2), alpha=0.1)
    assert_allclose(ci_es.eta2, (lb_eta2, ub_eta2), rtol=0.0001)
    assert_allclose(ci_es.ci_omega2, (lb_omega2, ub_omega2), rtol=0.025)