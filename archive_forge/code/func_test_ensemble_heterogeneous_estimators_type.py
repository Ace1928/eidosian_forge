import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.datasets import (
from sklearn.ensemble import (
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
@pytest.mark.parametrize('Ensemble', [VotingClassifier, StackingRegressor, VotingRegressor])
def test_ensemble_heterogeneous_estimators_type(Ensemble):
    if issubclass(Ensemble, ClassifierMixin):
        X, y = make_classification(n_samples=10)
        estimators = [('lr', LinearRegression())]
        ensemble_type = 'classifier'
    else:
        X, y = make_regression(n_samples=10)
        estimators = [('lr', LogisticRegression())]
        ensemble_type = 'regressor'
    ensemble = Ensemble(estimators=estimators)
    err_msg = 'should be a {}'.format(ensemble_type)
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)