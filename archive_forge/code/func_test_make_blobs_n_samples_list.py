import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_blobs_n_samples_list():
    n_samples = [50, 30, 20]
    X, y = make_blobs(n_samples=n_samples, n_features=2, random_state=0)
    assert X.shape == (sum(n_samples), 2), 'X shape mismatch'
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples), 'Incorrect number of samples per blob'