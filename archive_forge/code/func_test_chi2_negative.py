import warnings
import numpy as np
import pytest
import scipy.stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection._univariate_selection import _chisquare
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_chi2_negative(csr_container):
    X, y = ([[0, 1], [-1e-20, 1]], [0, 1])
    for X in (X, np.array(X), csr_container(X)):
        with pytest.raises(ValueError):
            chi2(X, y)