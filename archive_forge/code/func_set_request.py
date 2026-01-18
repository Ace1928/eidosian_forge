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
def set_request(estimator, method_name):
    if is_classifier(estimator) and method_name == 'partial_fit':
        estimator.set_partial_fit_request(classes=True)