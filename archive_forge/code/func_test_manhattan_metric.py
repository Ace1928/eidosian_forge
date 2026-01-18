import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_manhattan_metric(csr_container):
    X_csr = csr_container(X)
    clf = NearestCentroid(metric='manhattan')
    clf.fit(X, y)
    dense_centroid = clf.centroids_
    clf.fit(X_csr, y)
    assert_array_equal(clf.centroids_, dense_centroid)
    assert_array_equal(dense_centroid, [[-1, -1], [1, 1]])