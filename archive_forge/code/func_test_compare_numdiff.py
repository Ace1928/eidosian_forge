from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
@pytest.mark.slow
@pytest.mark.parametrize('use_sqrt', [False, True])
@pytest.mark.parametrize('reml', [False, True])
@pytest.mark.parametrize('profile_fe', [False, True])
def test_compare_numdiff(self, use_sqrt, reml, profile_fe):
    n_grp = 200
    grpsize = 5
    k_fe = 3
    k_re = 2
    np.random.seed(3558)
    exog_fe = np.random.normal(size=(n_grp * grpsize, k_fe))
    exog_re = np.random.normal(size=(n_grp * grpsize, k_re))
    exog_re[:, 0] = 1
    exog_vc = np.random.normal(size=(n_grp * grpsize, 3))
    slopes = np.random.normal(size=(n_grp, k_re))
    slopes[:, -1] *= 2
    slopes = np.kron(slopes, np.ones((grpsize, 1)))
    slopes_vc = np.random.normal(size=(n_grp, 3))
    slopes_vc = np.kron(slopes_vc, np.ones((grpsize, 1)))
    slopes_vc[:, -1] *= 2
    re_values = (slopes * exog_re).sum(1)
    vc_values = (slopes_vc * exog_vc).sum(1)
    err = np.random.normal(size=n_grp * grpsize)
    endog = exog_fe.sum(1) + re_values + vc_values + err
    groups = np.kron(range(n_grp), np.ones(grpsize))
    vc = {'a': {}, 'b': {}}
    for i in range(n_grp):
        ix = np.flatnonzero(groups == i)
        vc['a'][i] = exog_vc[ix, 0:2]
        vc['b'][i] = exog_vc[ix, 2:3]
    with pytest.warns(UserWarning, match='Using deprecated variance'):
        model = MixedLM(endog, exog_fe, groups, exog_re, exog_vc=vc, use_sqrt=use_sqrt)
    rslt = model.fit(reml=reml)
    loglike = loglike_function(model, profile_fe=profile_fe, has_fe=not profile_fe)
    try:
        for kr in range(5):
            fe_params = np.random.normal(size=k_fe)
            cov_re = np.random.normal(size=(k_re, k_re))
            cov_re = np.dot(cov_re.T, cov_re)
            vcomp = np.random.normal(size=2) ** 2
            params = MixedLMParams.from_components(fe_params, cov_re=cov_re, vcomp=vcomp)
            params_vec = params.get_packed(has_fe=not profile_fe, use_sqrt=use_sqrt)
            gr = -model.score(params, profile_fe=profile_fe)
            ngr = nd.approx_fprime(params_vec, loglike)
            assert_allclose(gr, ngr, rtol=0.001)
        if profile_fe is False and use_sqrt is False:
            hess, sing = model.hessian(rslt.params_object)
            if sing:
                pytest.fail('hessian should not be singular')
            hess *= -1
            params_vec = rslt.params_object.get_packed(use_sqrt=False, has_fe=True)
            loglike_h = loglike_function(model, profile_fe=False, has_fe=True)
            nhess = nd.approx_hess(params_vec, loglike_h)
            assert_allclose(hess, nhess, rtol=0.001)
    except AssertionError:
        if PLATFORM_OSX:
            pytest.xfail('fails on OSX due to unresolved numerical differences')
        else:
            raise