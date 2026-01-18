import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_kernel_centerer_feature_names_out():
    """Test that kernel centerer `feature_names_out`."""
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 4))
    X_pairwise = linear_kernel(X)
    centerer = KernelCenterer().fit(X_pairwise)
    names_out = centerer.get_feature_names_out()
    samples_out2 = X_pairwise.shape[1]
    assert_array_equal(names_out, [f'kernelcenterer{i}' for i in range(samples_out2)])