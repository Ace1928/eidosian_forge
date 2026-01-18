import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.datasets import (
from sklearn.ensemble import (
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
@pytest.mark.parametrize('X, y, estimator', [(*make_classification(n_samples=10), StackingClassifier(estimators=[('lr', LogisticRegression()), ('svm', LinearSVC(dual='auto')), ('rf', RandomForestClassifier(n_estimators=5, max_depth=3))], cv=2)), (*make_classification(n_samples=10), VotingClassifier(estimators=[('lr', LogisticRegression()), ('svm', LinearSVC(dual='auto')), ('rf', RandomForestClassifier(n_estimators=5, max_depth=3))])), (*make_regression(n_samples=10), StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto')), ('rf', RandomForestRegressor(n_estimators=5, max_depth=3))], cv=2)), (*make_regression(n_samples=10), VotingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto')), ('rf', RandomForestRegressor(n_estimators=5, max_depth=3))]))], ids=['stacking-classifier', 'voting-classifier', 'stacking-regressor', 'voting-regressor'])
def test_ensemble_heterogeneous_estimators_behavior(X, y, estimator):
    assert 'svm' in estimator.named_estimators
    assert estimator.named_estimators.svm is estimator.estimators[1][1]
    assert estimator.named_estimators.svm is estimator.named_estimators['svm']
    estimator.fit(X, y)
    assert len(estimator.named_estimators) == 3
    assert len(estimator.named_estimators_) == 3
    assert sorted(list(estimator.named_estimators_.keys())) == sorted(['lr', 'svm', 'rf'])
    estimator_new_params = clone(estimator)
    svm_estimator = SVC() if is_classifier(estimator) else SVR()
    estimator_new_params.set_params(svm=svm_estimator).fit(X, y)
    assert not hasattr(estimator_new_params, 'svm')
    assert estimator_new_params.named_estimators.lr.get_params() == estimator.named_estimators.lr.get_params()
    assert estimator_new_params.named_estimators.rf.get_params() == estimator.named_estimators.rf.get_params()
    estimator_dropped = clone(estimator)
    estimator_dropped.set_params(svm='drop')
    estimator_dropped.fit(X, y)
    assert len(estimator_dropped.named_estimators) == 3
    assert estimator_dropped.named_estimators.svm == 'drop'
    assert len(estimator_dropped.named_estimators_) == 3
    assert sorted(list(estimator_dropped.named_estimators_.keys())) == sorted(['lr', 'svm', 'rf'])
    for sub_est in estimator_dropped.named_estimators_:
        assert not isinstance(sub_est, type(estimator.named_estimators.svm))
    estimator.set_params(svm__C=10.0)
    estimator.set_params(rf__max_depth=5)
    assert estimator.get_params()['svm__C'] == estimator.get_params()['svm'].get_params()['C']
    assert estimator.get_params()['rf__max_depth'] == estimator.get_params()['rf'].get_params()['max_depth']