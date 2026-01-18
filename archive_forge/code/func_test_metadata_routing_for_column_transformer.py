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
@pytest.mark.parametrize('method', ['transform', 'fit_transform', 'fit'])
def test_metadata_routing_for_column_transformer(method):
    """Test that metadata is routed correctly for column transformer."""
    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]
    registry = _Registry()
    sample_weight, metadata = ([1], 'a')
    trs = ColumnTransformer([('trans', ConsumingTransformer(registry=registry).set_fit_request(sample_weight=True, metadata=True).set_transform_request(sample_weight=True, metadata=True), [0])])
    if method == 'transform':
        trs.fit(X, y)
        trs.transform(X, sample_weight=sample_weight, metadata=metadata)
    else:
        getattr(trs, method)(X, y, sample_weight=sample_weight, metadata=metadata)
    assert len(registry)
    for _trs in registry:
        check_recorded_metadata(obj=_trs, method=method, sample_weight=sample_weight, metadata=metadata)