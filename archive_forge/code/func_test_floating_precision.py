import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
@pytest.mark.parametrize('method', sorted(multitest_methods_names))
def test_floating_precision(method):
    pvals = np.full(6000, 0.99)
    pvals[0] = 1.138569e-56
    assert multipletests(pvals, method=method)[1][0] > 1e-60