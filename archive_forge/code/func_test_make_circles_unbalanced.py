import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_circles_unbalanced():
    X, y = make_circles(n_samples=(2, 8))
    assert np.sum(y == 0) == 2, 'Number of samples in inner circle is wrong'
    assert np.sum(y == 1) == 8, 'Number of samples in outer circle is wrong'
    assert X.shape == (10, 2), 'X shape mismatch'
    assert y.shape == (10,), 'y shape mismatch'
    with pytest.raises(ValueError, match='When a tuple, n_samples must have exactly two elements.'):
        make_circles(n_samples=(10,))