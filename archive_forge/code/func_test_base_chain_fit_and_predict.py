import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('response_method', ['predict_proba', 'predict_log_proba'])
def test_base_chain_fit_and_predict(response_method):
    X, Y = generate_multilabel_dataset_with_correlations()
    chains = [RegressorChain(Ridge()), ClassifierChain(LogisticRegression())]
    for chain in chains:
        chain.fit(X, Y)
        Y_pred = chain.predict(X)
        assert Y_pred.shape == Y.shape
        assert [c.coef_.size for c in chain.estimators_] == list(range(X.shape[1], X.shape[1] + Y.shape[1]))
    Y_prob = getattr(chains[1], response_method)(X)
    if response_method == 'predict_log_proba':
        Y_prob = np.exp(Y_prob)
    Y_binary = Y_prob >= 0.5
    assert_array_equal(Y_binary, Y_pred)
    assert isinstance(chains[1], ClassifierMixin)