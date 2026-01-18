import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore:Support for distance metrics:FutureWarning:sklearn')
def test_iris_shrinkage():
    for metric in ('euclidean', 'cosine'):
        for shrink_threshold in [None, 0.1, 0.5]:
            clf = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
            clf = clf.fit(iris.data, iris.target)
            score = np.mean(clf.predict(iris.data) == iris.target)
            assert score > 0.8, 'Failed with score = ' + str(score)