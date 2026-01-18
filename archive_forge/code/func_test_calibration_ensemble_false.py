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
@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
def test_calibration_ensemble_false(data, method):
    X, y = data
    clf = LinearSVC(dual='auto', random_state=7)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=3, ensemble=False)
    cal_clf.fit(X, y)
    cal_probas = cal_clf.predict_proba(X)
    unbiased_preds = cross_val_predict(clf, X, y, cv=3, method='decision_function')
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    else:
        calibrator = _SigmoidCalibration()
    calibrator.fit(unbiased_preds, y)
    clf.fit(X, y)
    clf_df = clf.decision_function(X)
    manual_probas = calibrator.predict(clf_df)
    assert_allclose(cal_probas[:, 1], manual_probas)