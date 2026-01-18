from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
def test_mcnemar_vectorized(reset_randomstate):
    ttk = np.random.randint(5, 15, size=(2, 2, 3))
    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=False)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=False) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)
    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=False, correction=False)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=False, correction=False) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)
    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=True)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=True) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)