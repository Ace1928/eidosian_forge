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
def test_discretenb_prior(DiscreteNaiveBayes, global_random_seed):
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    clf = DiscreteNaiveBayes().fit(X2, y2)
    assert_array_almost_equal(np.log(np.array([2, 2, 2]) / 6.0), clf.class_log_prior_, 8)