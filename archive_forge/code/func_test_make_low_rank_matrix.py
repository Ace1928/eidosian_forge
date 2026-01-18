import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_low_rank_matrix():
    X = make_low_rank_matrix(n_samples=50, n_features=25, effective_rank=5, tail_strength=0.01, random_state=0)
    assert X.shape == (50, 25), 'X shape mismatch'
    from numpy.linalg import svd
    u, s, v = svd(X)
    assert sum(s) - 5 < 0.1, 'X rank is not approximately 5'