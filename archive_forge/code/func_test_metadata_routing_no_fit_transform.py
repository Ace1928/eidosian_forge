import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.usefixtures('enable_slep006')
def test_metadata_routing_no_fit_transform():
    """Test metadata routing when the sub-estimator doesn't implement
    ``fit_transform``."""

    class NoFitTransform(BaseEstimator):

        def fit(self, X, y=None, sample_weight=None, metadata=None):
            assert sample_weight
            assert metadata
            return self

        def transform(self, X, sample_weight=None, metadata=None):
            assert sample_weight
            assert metadata
            return X
    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]
    _Registry()
    sample_weight, metadata = ([1], 'a')
    trs = ColumnTransformer([('trans', NoFitTransform().set_fit_request(sample_weight=True, metadata=True).set_transform_request(sample_weight=True, metadata=True), [0])])
    trs.fit(X, y, sample_weight=sample_weight, metadata=metadata)
    trs.fit_transform(X, y, sample_weight=sample_weight, metadata=metadata)