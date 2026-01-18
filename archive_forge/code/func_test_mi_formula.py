import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def test_mi_formula():
    np.random.seed(414)
    x = np.random.normal(size=(200, 4))
    x[[1, 3, 9], 0] = np.nan
    x[[1, 4, 3], 1] = np.nan
    x[[2, 11, 21], 2] = np.nan
    x[[11, 22, 99], 3] = np.nan
    df = pd.DataFrame({'y': x[:, 0], 'x1': x[:, 1], 'x2': x[:, 2], 'x3': x[:, 3]})
    fml = 'y ~ 0 + x1 + x2 + x3'

    def model_kwds_fn(x):
        return {'data': x}
    np.random.seed(2342)
    imp = BayesGaussMI(df.copy())
    mi = MI(imp, sm.OLS, formula=fml, burn=0, model_kwds_fn=model_kwds_fn)
    results_cb = lambda x: x
    r = mi.fit(results_cb=results_cb)
    r.summary()
    assert_allclose(r.params, np.r_[-0.05347919, -0.02479701, 0.10075517], 0.25, 0)
    c = np.asarray([[0.00418232, 0.00029746, -0.00035057], [0.00029746, 0.00407264, 0.00019496], [-0.00035057, 0.00019496, 0.00509413]])
    assert_allclose(r.cov_params(), c, 0.3, 0)
    assert_equal(len(r.results), 20)