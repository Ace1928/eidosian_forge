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
def test_fit_and_score_working():
    X, y = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel='linear', random_state=0)
    train, test = next(ShuffleSplit().split(X))
    fit_and_score_args = dict(estimator=clf, X=X, y=y, scorer=dict(), train=train, test=test, verbose=0, parameters={'max_iter': 100, 'tol': 0.1}, fit_params=None, score_params=None, return_parameters=True)
    result = _fit_and_score(**fit_and_score_args)
    assert result['parameters'] == fit_and_score_args['parameters']