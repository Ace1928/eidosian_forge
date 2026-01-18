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
def test_metaestimator_warnings():

    class WeightedMetaRegressorWarn(WeightedMetaRegressor):
        __metadata_request__fit = {'sample_weight': metadata_routing.WARN}
    with pytest.warns(UserWarning, match='Support for .* has recently been added to this class'):
        WeightedMetaRegressorWarn(estimator=LinearRegression().set_fit_request(sample_weight=False)).fit(X, y, sample_weight=my_weights)