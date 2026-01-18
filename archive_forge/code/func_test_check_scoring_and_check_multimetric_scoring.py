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
@pytest.mark.parametrize('scoring', (('accuracy',), ['precision'], {'acc': 'accuracy', 'precision': 'precision'}, ('accuracy', 'precision'), ['precision', 'accuracy'], {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}), ids=['single_tuple', 'single_list', 'dict_str', 'multi_tuple', 'multi_list', 'dict_callable'])
def test_check_scoring_and_check_multimetric_scoring(scoring):
    check_scoring_validator_for_single_metric_usecases(check_scoring)
    estimator = LinearSVC(dual='auto', random_state=0)
    estimator.fit([[1], [2], [3]], [1, 1, 0])
    scorers = _check_multimetric_scoring(estimator, scoring)
    assert isinstance(scorers, dict)
    assert sorted(scorers.keys()) == sorted(list(scoring))
    assert all([isinstance(scorer, _Scorer) for scorer in list(scorers.values())])
    assert all((scorer._response_method == 'predict' for scorer in scorers.values()))
    if 'acc' in scoring:
        assert_almost_equal(scorers['acc'](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0)
    if 'accuracy' in scoring:
        assert_almost_equal(scorers['accuracy'](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0)
    if 'precision' in scoring:
        assert_almost_equal(scorers['precision'](estimator, [[1], [2], [3]], [1, 0, 0]), 0.5)