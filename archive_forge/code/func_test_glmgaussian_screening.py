import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening
def test_glmgaussian_screening():
    y, x, idx_nonzero_true, beta = _get_gaussian_data()
    nobs = len(y)
    y = y - y.mean(0)
    screener_kwds = dict(pen_weight=nobs * 0.75, threshold_trim=0.001, ranking_attr='model.score_factor')
    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true, columns=['true'])
    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = GLMPenalized(y, xframe_true, family=family.Gaussian()).fit()
    parameters['oracle'] = res_oracle.params
    for k_keep in [1, 2]:
        mod_initial = GLMPenalized(y, x[:, :k_keep], family=family.Gaussian())
        screener = VariableScreening(mod_initial, **screener_kwds)
        exog_candidates = x[:, k_keep:]
        res_screen = screener.screen_exog(exog_candidates, maxiter=30)
        assert_equal(np.sort(res_screen.idx_nonzero), idx_nonzero_true)
        xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
        xnames[0] = 'const'
        res_screen.results_final.summary(xname=xnames)
        res_screen.results_pen.summary()
        assert_equal(res_screen.results_final.mle_retvals['converged'], True)
        ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
        parameters = parameters.join(ps, how='outer')
        assert_allclose(parameters['oracle'], parameters['final'], atol=1e-05)
        del parameters['final']