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
def test_assert_request_is_empty():
    requests = MetadataRequest(owner='test')
    assert_request_is_empty(requests)
    requests.fit.add_request(param='foo', alias=None)
    assert_request_is_empty(requests)
    requests.fit.add_request(param='bar', alias='value')
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests)
    assert_request_is_empty(requests, exclude='fit')
    requests.score.add_request(param='carrot', alias=True)
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests, exclude='fit')
    assert_request_is_empty(requests, exclude=['fit', 'score'])
    assert_request_is_empty(MetadataRouter(owner='test').add_self_request(WeightedMetaRegressor(estimator=None)).add(method_mapping='fit', estimator=ConsumingRegressor()))