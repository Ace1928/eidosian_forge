import warnings
import numpy as np
import pytest
import scipy.stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection._univariate_selection import _chisquare
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_chi2(csr_container):
    chi2 = mkchi2(k=1).fit(X, y)
    chi2 = mkchi2(k=1).fit(X, y)
    assert_array_equal(chi2.get_support(indices=True), [0])
    assert_array_equal(chi2.transform(X), np.array(X)[:, [0]])
    chi2 = mkchi2(k=2).fit(X, y)
    assert_array_equal(sorted(chi2.get_support(indices=True)), [0, 2])
    Xsp = csr_container(X, dtype=np.float64)
    chi2 = mkchi2(k=2).fit(Xsp, y)
    assert_array_equal(sorted(chi2.get_support(indices=True)), [0, 2])
    Xtrans = chi2.transform(Xsp)
    assert_array_equal(Xtrans.shape, [Xsp.shape[0], 2])
    Xtrans = Xtrans.toarray()
    Xtrans2 = mkchi2(k=2).fit_transform(Xsp, y).toarray()
    assert_array_almost_equal(Xtrans, Xtrans2)