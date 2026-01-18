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
def test_check_scoring_gridsearchcv():
    grid = GridSearchCV(LinearSVC(dual='auto'), param_grid={'C': [0.1, 1]}, cv=3)
    scorer = check_scoring(grid, scoring='f1')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    pipe = make_pipeline(LinearSVC(dual='auto'))
    scorer = check_scoring(pipe, scoring='f1')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    scores = cross_val_score(EstimatorWithFit(), [[1], [2], [3]], [1, 0, 1], scoring=DummyScorer(), cv=3)
    assert_array_equal(scores, 1)