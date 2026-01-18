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
def test_gnb_neg_priors():
    """Test whether an error is raised in case of negative priors"""
    clf = GaussianNB(priors=np.array([-1.0, 2.0]))
    msg = 'Priors must be non-negative'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)