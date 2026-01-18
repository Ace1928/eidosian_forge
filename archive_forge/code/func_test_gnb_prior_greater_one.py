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
def test_gnb_prior_greater_one():
    """Test if an error is raised if the sum of prior greater than one"""
    clf = GaussianNB(priors=np.array([2.0, 1.0]))
    msg = 'The sum of the priors should be 1'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)