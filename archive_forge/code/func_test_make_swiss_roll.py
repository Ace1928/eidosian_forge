import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('hole', [False, True])
def test_make_swiss_roll(hole):
    X, t = make_swiss_roll(n_samples=5, noise=0.0, random_state=0, hole=hole)
    assert X.shape == (5, 3)
    assert t.shape == (5,)
    assert_array_almost_equal(X[:, 0], t * np.cos(t))
    assert_array_almost_equal(X[:, 2], t * np.sin(t))