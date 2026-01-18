import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import (
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.validation import check_is_fitted
def test_no_metadata_always_works():
    """Test that when no metadata is passed, having a meta-estimator which does
    not yet support metadata routing works.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28246
    """

    class Estimator(_RoutingNotSupportedMixin, BaseEstimator):

        def fit(self, X, y, metadata=None):
            return self
    MetaRegressor(estimator=Estimator()).fit(X, y)
    with pytest.raises(NotImplementedError, match='Estimator has not implemented metadata routing yet.'):
        MetaRegressor(estimator=Estimator()).fit(X, y, metadata=my_groups)