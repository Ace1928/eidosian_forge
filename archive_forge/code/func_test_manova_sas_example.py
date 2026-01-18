import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant
def test_manova_sas_example():
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    r = mod.mv_test()
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Value'], 0.60143661, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Value'], 0.44702843, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc['Hotelling-Lawley trace', 'Value'], 0.58210348, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Value'], 0.3553089, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'F Value'], 0.77, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'F Value'], 0.86, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc['Hotelling-Lawley trace', 'F Value'], 0.75, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'F Value'], 1.07, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Num DF'], 6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Num DF'], 6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc['Hotelling-Lawley trace', 'Num DF'], 6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Num DF'], 3, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Den DF'], 16, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Den DF'], 18, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc['Hotelling-Lawley trace', 'Den DF'], 9.0909, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Den DF'], 9, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Pr > F'], 0.6032, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Pr > F'], 0.5397, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc['Hotelling-Lawley trace', 'Pr > F'], 0.6272, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Pr > F'], 0.4109, decimal=4)