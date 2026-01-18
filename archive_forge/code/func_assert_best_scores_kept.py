import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def assert_best_scores_kept(score_filter):
    scores = score_filter.scores_
    support = score_filter.get_support()
    assert_allclose(np.sort(scores[support]), np.sort(scores)[-support.sum():])