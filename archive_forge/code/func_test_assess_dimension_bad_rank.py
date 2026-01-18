import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_assess_dimension_bad_rank():
    spectrum = np.array([1, 1e-30, 1e-30, 1e-30])
    n_samples = 10
    for rank in (0, 5):
        with pytest.raises(ValueError, match='should be in \\[1, n_features - 1\\]'):
            _assess_dimension(spectrum, rank, n_samples)