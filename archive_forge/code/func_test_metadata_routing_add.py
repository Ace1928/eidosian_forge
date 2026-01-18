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
def test_metadata_routing_add():
    router = MetadataRouter(owner='test').add(method_mapping='fit', est=ConsumingRegressor().set_fit_request(sample_weight='weights'))
    assert str(router) == "{'est': {'mapping': [{'callee': 'fit', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': 'weights', 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': None}}}}"
    router = MetadataRouter(owner='test').add(method_mapping=MethodMapping().add(callee='score', caller='fit'), est=ConsumingRegressor().set_score_request(sample_weight=True))
    assert str(router) == "{'est': {'mapping': [{'callee': 'score', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': True}}}}"