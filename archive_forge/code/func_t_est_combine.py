import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
@pytest.mark.slow
def t_est_combine(self):
    gen = np.random.RandomState(3897)
    x1 = gen.normal(size=300)
    x2 = gen.normal(size=300)
    y = x1 + x2 + gen.normal(size=300)
    x1[0:100] = np.nan
    x2[250:] = np.nan
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    idata = mice.MICEData(df)
    mi = mice.MICE('y ~ x1 + x2', sm.OLS, idata, n_skip=20)
    result = mi.fit(10, 20)
    fmi = np.asarray([0.1778143, 0.11057262, 0.29626521])
    assert_allclose(result.frac_miss_info, fmi, atol=1e-05)
    params = np.asarray([-0.03486102, 0.96236808, 0.9970371])
    assert_allclose(result.params, params, atol=1e-05)
    tvalues = np.asarray([-0.54674776, 15.28091069, 13.61359403])
    assert_allclose(result.tvalues, tvalues, atol=1e-05)