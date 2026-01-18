import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
@pytest.mark.parametrize('case', models)
def test_distr(case):
    y, x = (y_count, x_const)
    nobs = len(y)
    np.random.seed(987456348)
    cls_model, kwds, params = case
    if issubclass(cls_model, BinaryModel):
        y = (y > 0.5).astype(float)
    mod = cls_model(y, x, **kwds)
    params_dgp = params
    distr = mod.get_distribution(params_dgp)
    assert distr.pmf(1).ndim == 1
    try:
        y2 = distr.rvs(size=(nobs, 1)).squeeze()
    except ValueError:
        y2 = distr.rvs(size=nobs).squeeze()
    mod = cls_model(y2, x, **kwds)
    res = mod.fit(start_params=params_dgp, method='bfgs', maxiter=500)
    distr2 = mod.get_distribution(res.params)
    assert_allclose(distr2.mean().squeeze()[0], y2.mean(), rtol=0.2)
    assert_allclose(distr2.var().squeeze()[0], y2.var(), rtol=0.2)
    var_ = res.predict(which='var')
    assert_allclose(var_, distr2.var().squeeze(), rtol=1e-12)
    mean = res.predict()
    assert_allclose(res.resid_pearson, (y2 - mean) / np.sqrt(var_), rtol=1e-13)
    if not issubclass(cls_model, BinaryModel):
        probs = res.predict(which='prob', y_values=np.arange(5))
        assert probs.shape == (len(mod.endog), 5)
        probs2 = res.get_prediction(which='prob', y_values=np.arange(5), average=True)
        assert_allclose(probs2.predicted, probs.mean(0), rtol=1e-10)
        dia = res.get_diagnostic()
        dia.probs_predicted
    if cls_model in models_influ:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            influ = res.get_influence()
            influ.summary_frame()
        assert influ.resid.shape == (len(y2),)
        try:
            resid = influ.resid_score_factor()
            assert resid.shape == (len(y2),)
        except AttributeError:
            pass
        resid = influ.resid_score()
        assert resid.shape == (len(y2),)
        f_sc = influ.d_fittedvalues_scaled
        assert f_sc.shape == (len(y2),)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                influ.plot_influence()
        except ImportError:
            pass