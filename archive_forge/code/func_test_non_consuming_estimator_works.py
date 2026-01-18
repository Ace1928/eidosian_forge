import copy
import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import (
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.experimental import (
from sklearn.feature_selection import (
from sklearn.impute import IterativeImputer
from sklearn.linear_model import (
from sklearn.model_selection import (
from sklearn.multiclass import (
from sklearn.multioutput import (
from sklearn.pipeline import FeatureUnion
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tests.metadata_routing_common import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.parametrize('metaestimator', METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_non_consuming_estimator_works(metaestimator):
    if 'estimator' not in metaestimator:
        return

    def set_request(estimator, method_name):
        if is_classifier(estimator) and method_name == 'partial_fit':
            estimator.set_partial_fit_request(classes=True)
    cls = metaestimator['metaestimator']
    X = metaestimator['X']
    y = metaestimator['y']
    routing_methods = metaestimator['estimator_routing_methods']
    for method_name in routing_methods:
        kwargs, (estimator, _), (_, _), (_, _) = get_init_args(metaestimator, sub_estimator_consumes=False)
        instance = cls(**kwargs)
        set_request(estimator, method_name)
        method = getattr(instance, method_name)
        extra_method_args = metaestimator.get('method_args', {}).get(method_name, {})
        method(X, y, **extra_method_args)