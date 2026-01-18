import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_n_samples_equal_n_components():
    ipca = IncrementalPCA(n_components=5)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ipca.partial_fit(np.random.randn(5, 7))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ipca.fit(np.random.randn(5, 7))