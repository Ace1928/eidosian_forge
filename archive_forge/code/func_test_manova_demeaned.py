import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant
def test_manova_demeaned():
    ng = 5
    loc = ['Basal', 'Occ', 'Max'] * ng
    y1 = (np.random.randn(ng, 3) + [0, 0.5, 1]).ravel()
    y2 = (np.random.randn(ng, 3) + [0.25, 0.75, 1]).ravel()
    y3 = (np.random.randn(ng, 3) + [0.3, 0.6, 1]).ravel()
    dta = pd.DataFrame(dict(Loc=loc, Basal=y1, Occ=y2, Max=y3))
    mod = MANOVA.from_formula('Basal + Occ + Max ~ C(Loc, Helmert)', data=dta)
    res1 = mod.mv_test()
    means = dta[['Basal', 'Occ', 'Max']].mean()
    dta[['Basal', 'Occ', 'Max']] = dta[['Basal', 'Occ', 'Max']] - means
    mod = MANOVA.from_formula('Basal + Occ + Max ~ C(Loc, Helmert)', data=dta)
    res2 = mod.mv_test(skip_intercept_test=True)
    stat1 = res1.results['C(Loc, Helmert)']['stat'].to_numpy(float)
    stat2 = res2.results['C(Loc, Helmert)']['stat'].to_numpy(float)
    assert_allclose(stat1, stat2, rtol=1e-10)