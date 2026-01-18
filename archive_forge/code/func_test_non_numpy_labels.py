import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
def test_non_numpy_labels():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    assert silhouette_score(list(X), list(y)) == silhouette_score(X, y)