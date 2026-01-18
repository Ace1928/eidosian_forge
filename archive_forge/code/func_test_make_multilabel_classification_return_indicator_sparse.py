import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_multilabel_classification_return_indicator_sparse():
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        X, Y = make_multilabel_classification(n_samples=25, n_features=20, n_classes=3, random_state=0, return_indicator='sparse', allow_unlabeled=allow_unlabeled)
        assert X.shape == (25, 20), 'X shape mismatch'
        assert Y.shape == (25, 3), 'Y shape mismatch'
        assert sp.issparse(Y)