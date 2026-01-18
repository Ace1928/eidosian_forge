import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
def check_scoring_validator_for_single_metric_usecases(scoring_validator):
    estimator = EstimatorWithFitAndScore()
    estimator.fit([[1]], [1])
    scorer = scoring_validator(estimator)
    assert isinstance(scorer, _PassthroughScorer)
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    pattern = "If no scoring is specified, the estimator passed should have a 'score' method\\. The estimator .* does not\\."
    with pytest.raises(TypeError, match=pattern):
        scoring_validator(estimator)
    scorer = scoring_validator(estimator, scoring='accuracy')
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    estimator = EstimatorWithFit()
    scorer = scoring_validator(estimator, scoring='accuracy')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    if scoring_validator is check_scoring:
        estimator = EstimatorWithFit()
        scorer = scoring_validator(estimator, allow_none=True)
        assert scorer is None