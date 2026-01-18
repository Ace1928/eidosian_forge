import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_additive_chi2_sampler(csr_container):
    X_ = X[:, np.newaxis, :]
    Y_ = Y[np.newaxis, :, :]
    large_kernel = 2 * X_ * Y_ / (X_ + Y_)
    kernel = large_kernel.sum(axis=2)
    transform = AdditiveChi2Sampler(sample_steps=3)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    assert_array_almost_equal(kernel, kernel_approx, 1)
    X_sp_trans = transform.fit_transform(csr_container(X))
    Y_sp_trans = transform.transform(csr_container(Y))
    assert_array_equal(X_trans, X_sp_trans.toarray())
    assert_array_equal(Y_trans, Y_sp_trans.toarray())
    Y_neg = Y.copy()
    Y_neg[0, 0] = -1
    msg = 'Negative values in data passed to'
    with pytest.raises(ValueError, match=msg):
        transform.fit(Y_neg)