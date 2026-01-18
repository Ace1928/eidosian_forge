import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_feature_names_out():
    """Check feature names out for IncrementalPCA."""
    ipca = IncrementalPCA(n_components=2).fit(iris.data)
    names = ipca.get_feature_names_out()
    assert_array_equal([f'incrementalpca{i}' for i in range(2)], names)