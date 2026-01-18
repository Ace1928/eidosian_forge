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
def test_gnb_wrong_nb_priors():
    """Test whether an error is raised if the number of prior is different
    from the number of class"""
    clf = GaussianNB(priors=np.array([0.25, 0.25, 0.25, 0.25]))
    msg = 'Number of priors must match number of classes'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)