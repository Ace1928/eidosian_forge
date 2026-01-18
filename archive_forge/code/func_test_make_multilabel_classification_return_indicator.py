import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_multilabel_classification_return_indicator():
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        X, Y = make_multilabel_classification(n_samples=25, n_features=20, n_classes=3, random_state=0, allow_unlabeled=allow_unlabeled)
        assert X.shape == (25, 20), 'X shape mismatch'
        assert Y.shape == (25, 3), 'Y shape mismatch'
        assert np.all(np.sum(Y, axis=0) > min_length)
    X2, Y2, p_c, p_w_c = make_multilabel_classification(n_samples=25, n_features=20, n_classes=3, random_state=0, allow_unlabeled=allow_unlabeled, return_distributions=True)
    assert_array_almost_equal(X, X2)
    assert_array_equal(Y, Y2)
    assert p_c.shape == (3,)
    assert_almost_equal(p_c.sum(), 1)
    assert p_w_c.shape == (20, 3)
    assert_almost_equal(p_w_c.sum(axis=0), [1] * 3)