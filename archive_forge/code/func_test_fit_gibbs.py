import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_fit_gibbs(csc_container):
    rng = np.random.RandomState(42)
    X = np.array([[0.0], [1.0]])
    rbm1 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
    rbm1.fit(X)
    assert_almost_equal(rbm1.components_, np.array([[0.02649814], [0.02009084]]), decimal=4)
    assert_almost_equal(rbm1.gibbs(X), X)
    rng = np.random.RandomState(42)
    X = csc_container([[0.0], [1.0]])
    rbm2 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
    rbm2.fit(X)
    assert_almost_equal(rbm2.components_, np.array([[0.02649814], [0.02009084]]), decimal=4)
    assert_almost_equal(rbm2.gibbs(X), X.toarray())
    assert_almost_equal(rbm1.components_, rbm2.components_)