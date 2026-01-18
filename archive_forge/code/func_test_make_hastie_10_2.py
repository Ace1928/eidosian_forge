import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_hastie_10_2():
    X, y = make_hastie_10_2(n_samples=100, random_state=0)
    assert X.shape == (100, 10), 'X shape mismatch'
    assert y.shape == (100,), 'y shape mismatch'
    assert np.unique(y).shape == (2,), 'Unexpected number of classes'