import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_array_almost_equal
def test_inverse_transform_Xred_deprecation():
    X = np.array([0, 0, 1]).reshape(1, 3)
    est = FeatureAgglomeration(n_clusters=1, pooling_func=np.mean)
    est.fit(X)
    Xt = est.transform(X)
    with pytest.raises(TypeError, match='Missing required positional argument'):
        est.inverse_transform()
    with pytest.raises(ValueError, match='Please provide only'):
        est.inverse_transform(Xt=Xt, Xred=Xt)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('error')
        est.inverse_transform(Xt)
    with pytest.warns(FutureWarning, match='Input argument `Xred` was renamed to `Xt`'):
        est.inverse_transform(Xred=Xt)