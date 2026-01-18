import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('strategy', ['stratified', 'most_frequent', 'prior', 'uniform', 'constant'])
def test_dtype_of_classifier_probas(strategy):
    y = [0, 2, 1, 1]
    X = np.zeros(4)
    model = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    probas = model.fit(X, y).predict_proba(X)
    assert probas.dtype == np.float64