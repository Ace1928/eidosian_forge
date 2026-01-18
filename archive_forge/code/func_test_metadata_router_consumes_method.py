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
def test_metadata_router_consumes_method():
    """Test that MetadataRouter().consumes method works as expected."""
    cases = [(WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight=True)), {'sample_weight'}, {'sample_weight'}), (WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='my_weights')), {'my_weights', 'sample_weight'}, {'my_weights'})]
    for obj, input, output in cases:
        assert obj.get_metadata_routing().consumes(method='fit', params=input) == output