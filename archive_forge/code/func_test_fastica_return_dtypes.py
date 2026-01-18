import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
def test_fastica_return_dtypes(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    k_, mixing_, s_ = fastica(X, max_iter=1000, whiten='unit-variance', random_state=rng)
    assert k_.dtype == global_dtype
    assert mixing_.dtype == global_dtype
    assert s_.dtype == global_dtype