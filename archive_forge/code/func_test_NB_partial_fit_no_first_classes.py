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
@pytest.mark.parametrize('NaiveBayes', ALL_NAIVE_BAYES_CLASSES)
def test_NB_partial_fit_no_first_classes(NaiveBayes, global_random_seed):
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    with pytest.raises(ValueError, match='classes must be passed on the first call to partial_fit.'):
        NaiveBayes().partial_fit(X2, y2)
    clf = NaiveBayes()
    clf.partial_fit(X2, y2, classes=np.unique(y2))
    with pytest.raises(ValueError, match='is not the same as on last call to partial_fit'):
        clf.partial_fit(X2, y2, classes=np.arange(42))