import functools
from inspect import signature
import numpy as np
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import (
from sklearn.utils.validation import check_is_fitted
@pytest.mark.parametrize('estimator', DATA_VALIDATION_META_ESTIMATORS, ids=_get_meta_estimator_id)
def test_meta_estimators_delegate_data_validation(estimator):
    rng = np.random.RandomState(0)
    set_random_state(estimator)
    n_samples = 30
    X = rng.choice(np.array(['aa', 'bb', 'cc'], dtype=object), size=n_samples)
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(3, size=n_samples)
    X = _enforce_estimator_tags_X(estimator, X).tolist()
    y = _enforce_estimator_tags_y(estimator, y).tolist()
    estimator.fit(X, y)
    assert not hasattr(estimator, 'n_features_in_')