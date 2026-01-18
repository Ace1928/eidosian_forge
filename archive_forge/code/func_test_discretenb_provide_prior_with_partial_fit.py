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
def test_discretenb_provide_prior_with_partial_fit(DiscreteNaiveBayes):
    iris = load_iris()
    iris_data1, iris_data2, iris_target1, iris_target2 = train_test_split(iris.data, iris.target, test_size=0.4, random_state=415)
    for prior in [None, [0.3, 0.3, 0.4]]:
        clf_full = DiscreteNaiveBayes(class_prior=prior)
        clf_full.fit(iris.data, iris.target)
        clf_partial = DiscreteNaiveBayes(class_prior=prior)
        clf_partial.partial_fit(iris_data1, iris_target1, classes=[0, 1, 2])
        clf_partial.partial_fit(iris_data2, iris_target2)
        assert_array_almost_equal(clf_full.class_log_prior_, clf_partial.class_log_prior_)