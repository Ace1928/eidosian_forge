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
def get_init_args(metaestimator_info, sub_estimator_consumes):
    """Get the init args for a metaestimator

    This is a helper function to get the init args for a metaestimator from
    the METAESTIMATORS list. It returns an empty dict if no init args are
    required.

    Parameters
    ----------
    metaestimator_info : dict
        The metaestimator info from METAESTIMATORS

    sub_estimator_consumes : bool
        Whether the sub-estimator consumes metadata or not.

    Returns
    -------
    kwargs : dict
        The init args for the metaestimator.

    (estimator, estimator_registry) : (estimator, registry)
        The sub-estimator and the corresponding registry.

    (scorer, scorer_registry) : (scorer, registry)
        The scorer and the corresponding registry.

    (cv, cv_registry) : (CV splitter, registry)
        The CV splitter and the corresponding registry.
    """
    kwargs = metaestimator_info.get('init_args', {})
    estimator, estimator_registry = (None, None)
    scorer, scorer_registry = (None, None)
    cv, cv_registry = (None, None)
    if 'estimator' in metaestimator_info:
        estimator_name = metaestimator_info['estimator_name']
        estimator_registry = _Registry()
        sub_estimator_type = metaestimator_info['estimator']
        if sub_estimator_consumes:
            if sub_estimator_type == 'regressor':
                estimator = ConsumingRegressor(estimator_registry)
            else:
                estimator = ConsumingClassifier(estimator_registry)
        elif sub_estimator_type == 'regressor':
            estimator = NonConsumingRegressor()
        else:
            estimator = NonConsumingClassifier()
        kwargs[estimator_name] = estimator
    if 'scorer_name' in metaestimator_info:
        scorer_name = metaestimator_info['scorer_name']
        scorer_registry = _Registry()
        scorer = ConsumingScorer(registry=scorer_registry)
        kwargs[scorer_name] = scorer
    if 'cv_name' in metaestimator_info:
        cv_name = metaestimator_info['cv_name']
        cv_registry = _Registry()
        cv = ConsumingSplitter(registry=cv_registry)
        kwargs[cv_name] = cv
    return (kwargs, (estimator, estimator_registry), (scorer, scorer_registry), (cv, cv_registry))