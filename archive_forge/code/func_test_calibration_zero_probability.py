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
def test_calibration_zero_probability():

    class ZeroCalibrator:

        def predict(self, X):
            return np.zeros(X.shape[0])
    X, y = make_blobs(n_samples=50, n_features=10, random_state=7, centers=10, cluster_std=15.0)
    clf = DummyClassifier().fit(X, y)
    calibrator = ZeroCalibrator()
    cal_clf = _CalibratedClassifier(estimator=clf, calibrators=[calibrator], classes=clf.classes_)
    probas = cal_clf.predict_proba(X)
    assert_allclose(probas, 1.0 / clf.n_classes_)