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
def check_cross_validate_single_metric(clf, X, y, scores, cv):
    train_mse_scores, test_mse_scores, train_r2_scores, test_r2_scores, fitted_estimators = scores
    for return_train_score, dict_len in ((True, 4), (False, 3)):
        if return_train_score:
            mse_scores_dict = cross_validate(clf, X, y, scoring='neg_mean_squared_error', return_train_score=True, cv=cv)
            assert_array_almost_equal(mse_scores_dict['train_score'], train_mse_scores)
        else:
            mse_scores_dict = cross_validate(clf, X, y, scoring='neg_mean_squared_error', return_train_score=False, cv=cv)
        assert isinstance(mse_scores_dict, dict)
        assert len(mse_scores_dict) == dict_len
        assert_array_almost_equal(mse_scores_dict['test_score'], test_mse_scores)
        if return_train_score:
            r2_scores_dict = cross_validate(clf, X, y, scoring=['r2'], return_train_score=True, cv=cv)
            assert_array_almost_equal(r2_scores_dict['train_r2'], train_r2_scores, True)
        else:
            r2_scores_dict = cross_validate(clf, X, y, scoring=['r2'], return_train_score=False, cv=cv)
        assert isinstance(r2_scores_dict, dict)
        assert len(r2_scores_dict) == dict_len
        assert_array_almost_equal(r2_scores_dict['test_r2'], test_r2_scores)
    mse_scores_dict = cross_validate(clf, X, y, scoring='neg_mean_squared_error', return_estimator=True, cv=cv)
    for k, est in enumerate(mse_scores_dict['estimator']):
        est_coef = est.coef_.copy()
        if issparse(est_coef):
            est_coef = est_coef.toarray()
        fitted_est_coef = fitted_estimators[k].coef_.copy()
        if issparse(fitted_est_coef):
            fitted_est_coef = fitted_est_coef.toarray()
        assert_almost_equal(est_coef, fitted_est_coef)
        assert_almost_equal(est.intercept_, fitted_estimators[k].intercept_)