import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import (
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
def test_calibration_dict_pipeline(dict_data, dict_data_pipeline):
    """Test that calibration works in prefit pipeline with transformer

    `X` is not array-like, sparse matrix or dataframe at the start.
    See https://github.com/scikit-learn/scikit-learn/issues/8710

    Also test it can predict without running into validation errors.
    See https://github.com/scikit-learn/scikit-learn/issues/19637
    """
    X, y = dict_data
    clf = dict_data_pipeline
    calib_clf = CalibratedClassifierCV(clf, cv='prefit')
    calib_clf.fit(X, y)
    assert_array_equal(calib_clf.classes_, clf.classes_)
    assert not hasattr(clf, 'n_features_in_')
    assert not hasattr(calib_clf, 'n_features_in_')
    calib_clf.predict(X)
    calib_clf.predict_proba(X)