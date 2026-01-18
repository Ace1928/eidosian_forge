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
def test_get_metadata_routing():

    class TestDefaultsBadMethodName(_MetadataRequester):
        __metadata_request__fit = {'sample_weight': None, 'my_param': None}
        __metadata_request__score = {'sample_weight': None, 'my_param': True, 'my_other_param': None}
        __metadata_request__other_method = {'my_param': True}

    class TestDefaults(_MetadataRequester):
        __metadata_request__fit = {'sample_weight': None, 'my_other_param': None}
        __metadata_request__score = {'sample_weight': None, 'my_param': True, 'my_other_param': None}
        __metadata_request__predict = {'my_param': True}
    with pytest.raises(AttributeError, match="'MetadataRequest' object has no attribute 'other_method'"):
        TestDefaultsBadMethodName().get_metadata_routing()
    expected = {'score': {'my_param': True, 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': None}, 'predict': {'my_param': True}}
    assert_request_equal(TestDefaults().get_metadata_routing(), expected)
    est = TestDefaults().set_score_request(my_param='other_param')
    expected = {'score': {'my_param': 'other_param', 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': None}, 'predict': {'my_param': True}}
    assert_request_equal(est.get_metadata_routing(), expected)
    est = TestDefaults().set_fit_request(sample_weight=True)
    expected = {'score': {'my_param': True, 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': True}, 'predict': {'my_param': True}}
    assert_request_equal(est.get_metadata_routing(), expected)