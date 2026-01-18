import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('DiscreteNaiveBayes', DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_provide_prior(DiscreteNaiveBayes):
    clf = DiscreteNaiveBayes(class_prior=[0.5, 0.5])
    clf.fit([[0], [0], [1]], [0, 0, 1])
    prior = np.exp(clf.class_log_prior_)
    assert_array_almost_equal(prior, np.array([0.5, 0.5]))
    msg = 'Number of priors must match number of classes'
    with pytest.raises(ValueError, match=msg):
        clf.fit([[0], [1], [2]], [0, 1, 2])
    msg = 'is not the same as on last call to partial_fit'
    with pytest.raises(ValueError, match=msg):
        clf.partial_fit([[0], [1]], [0, 1], classes=[0, 1, 1])