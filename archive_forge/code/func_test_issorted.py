import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
@pytest.mark.parametrize('method', sorted(multitest_methods_names))
def test_issorted(method):
    pvals = np.array([31, 9958111, 7430818, 8653643, 9892855, 876, 2651691, 145836, 9931, 6174747]) * 1e-07
    sortind = np.argsort(pvals)
    sortrevind = sortind.argsort()
    pvals_sorted = pvals[sortind]
    res1 = multipletests(pvals, method=method, is_sorted=False)
    res2 = multipletests(pvals_sorted, method=method, is_sorted=True)
    assert_equal(res2[0][sortrevind], res1[0])
    assert_allclose(res2[0][sortrevind], res1[0], rtol=1e-10)