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
def test_nested_routing_conflict():
    pipeline = SimplePipeline([MetaTransformer(transformer=ConsumingTransformer().set_fit_request(metadata=True, sample_weight=False).set_transform_request(sample_weight=True)), WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight=True)).set_fit_request(sample_weight='outer_weights')])
    w1, w2 = ([1], [2])
    with pytest.raises(ValueError, match=re.escape('In WeightedMetaRegressor, there is a conflict on sample_weight between what is requested for this estimator and what is requested by its children. You can resolve this conflict by using an alias for the child estimator(s) requested metadata.')):
        pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2)