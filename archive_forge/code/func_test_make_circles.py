import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_circles():
    factor = 0.3
    for n_samples, n_outer, n_inner in [(7, 3, 4), (8, 4, 4)]:
        X, y = make_circles(n_samples, shuffle=False, noise=None, factor=factor)
        assert X.shape == (n_samples, 2), 'X shape mismatch'
        assert y.shape == (n_samples,), 'y shape mismatch'
        center = [0.0, 0.0]
        for x, label in zip(X, y):
            dist_sqr = ((x - center) ** 2).sum()
            dist_exp = 1.0 if label == 0 else factor ** 2
            dist_exp = 1.0 if label == 0 else factor ** 2
            assert_almost_equal(dist_sqr, dist_exp, err_msg='Point is not on expected circle')
        assert X[y == 0].shape == (n_outer, 2), 'Samples not correctly distributed across circles.'
        assert X[y == 1].shape == (n_inner, 2), 'Samples not correctly distributed across circles.'