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
def test_learning_curve_incremental_learning_fit_params():
    X, y = make_classification(n_samples=30, n_features=1, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=0)
    estimator = MockIncrementalImprovingEstimator(20, ['sample_weight'])
    err_msg = "Expected fit parameter\\(s\\) \\['sample_weight'\\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        learning_curve(estimator, X, y, cv=3, exploit_incremental_learning=True, train_sizes=np.linspace(0.1, 1.0, 10), error_score='raise')
    err_msg = 'Fit parameter sample_weight has length 3; expected'
    with pytest.raises(AssertionError, match=err_msg):
        learning_curve(estimator, X, y, cv=3, exploit_incremental_learning=True, train_sizes=np.linspace(0.1, 1.0, 10), error_score='raise', fit_params={'sample_weight': np.ones(3)})
    learning_curve(estimator, X, y, cv=3, exploit_incremental_learning=True, train_sizes=np.linspace(0.1, 1.0, 10), error_score='raise', fit_params={'sample_weight': np.ones(2)})