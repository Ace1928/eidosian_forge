import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_multilabel_classification_return_sequences():
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        X, Y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=3, random_state=0, return_indicator=False, allow_unlabeled=allow_unlabeled)
        assert X.shape == (100, 20), 'X shape mismatch'
        if not allow_unlabeled:
            assert max([max(y) for y in Y]) == 2
        assert min([len(y) for y in Y]) == min_length
        assert max([len(y) for y in Y]) <= 3