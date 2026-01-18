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
def test_discretenb_partial_fit(DiscreteNaiveBayes):
    clf1 = DiscreteNaiveBayes()
    clf1.fit([[0, 1], [1, 0], [1, 1]], [0, 1, 1])
    clf2 = DiscreteNaiveBayes()
    clf2.partial_fit([[0, 1], [1, 0], [1, 1]], [0, 1, 1], classes=[0, 1])
    assert_array_equal(clf1.class_count_, clf2.class_count_)
    if DiscreteNaiveBayes is CategoricalNB:
        for i in range(len(clf1.category_count_)):
            assert_array_equal(clf1.category_count_[i], clf2.category_count_[i])
    else:
        assert_array_equal(clf1.feature_count_, clf2.feature_count_)
    clf3 = DiscreteNaiveBayes()
    clf3.partial_fit([[0, 1]], [0], classes=[0, 1])
    clf3.partial_fit([[1, 0]], [1])
    clf3.partial_fit([[1, 1]], [1])
    assert_array_equal(clf1.class_count_, clf3.class_count_)
    if DiscreteNaiveBayes is CategoricalNB:
        for i in range(len(clf1.category_count_)):
            assert_array_equal(clf1.category_count_[i].shape, clf3.category_count_[i].shape)
            assert_array_equal(np.sum(clf1.category_count_[i], axis=1), np.sum(clf3.category_count_[i], axis=1))
        assert_array_equal(clf1.category_count_[0][0], np.array([1, 0]))
        assert_array_equal(clf1.category_count_[0][1], np.array([0, 2]))
        assert_array_equal(clf1.category_count_[1][0], np.array([0, 1]))
        assert_array_equal(clf1.category_count_[1][1], np.array([1, 1]))
    else:
        assert_array_equal(clf1.feature_count_, clf3.feature_count_)