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
def test_method_metadata_request():
    mmr = MethodMetadataRequest(owner='test', method='fit')
    with pytest.raises(ValueError, match="The alias you're setting for"):
        mmr.add_request(param='foo', alias=1.4)
    mmr.add_request(param='foo', alias=None)
    assert mmr.requests == {'foo': None}
    mmr.add_request(param='foo', alias=False)
    assert mmr.requests == {'foo': False}
    mmr.add_request(param='foo', alias=True)
    assert mmr.requests == {'foo': True}
    mmr.add_request(param='foo', alias='foo')
    assert mmr.requests == {'foo': True}
    mmr.add_request(param='foo', alias='bar')
    assert mmr.requests == {'foo': 'bar'}
    assert mmr._get_param_names(return_alias=False) == {'foo'}
    assert mmr._get_param_names(return_alias=True) == {'bar'}