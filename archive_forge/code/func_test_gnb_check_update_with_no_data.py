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
def test_gnb_check_update_with_no_data():
    """Test when the partial fit is called without any data"""
    prev_points = 100
    mean = 0.0
    var = 1.0
    x_empty = np.empty((0, X.shape[1]))
    tmean, tvar = GaussianNB._update_mean_variance(prev_points, mean, var, x_empty)
    assert tmean == mean
    assert tvar == var