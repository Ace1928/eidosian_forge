import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_learning_curve_batch_and_incremental_learning_are_equal():
    X, y = make_classification(n_samples=30, n_features=1, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=0)
    train_sizes = np.linspace(0.2, 1.0, 5)
    estimator = PassiveAggressiveClassifier(max_iter=1, tol=None, shuffle=False)
    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=3, exploit_incremental_learning=True)
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(estimator, X, y, cv=3, train_sizes=train_sizes, exploit_incremental_learning=False)
    assert_array_equal(train_sizes_inc, train_sizes_batch)
    assert_array_almost_equal(train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1))
    assert_array_almost_equal(test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1))