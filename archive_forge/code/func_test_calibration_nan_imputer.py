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
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration_nan_imputer(ensemble):
    """Test that calibration can accept nan"""
    X, y = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X[0, 0] = np.nan
    clf = Pipeline([('imputer', SimpleImputer()), ('rf', RandomForestClassifier(n_estimators=1))])
    clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic', ensemble=ensemble)
    clf_c.fit(X, y)
    clf_c.predict(X)