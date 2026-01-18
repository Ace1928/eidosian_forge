from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
@pytest.mark.parametrize('Est', (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize('params, expected_error_message', [({'resource': 'not_a_parameter'}, 'Cannot use resource=not_a_parameter which is not supported'), ({'resource': 'a', 'max_resources': 100}, 'Cannot use parameter a as the resource since it is part of'), ({'max_resources': 'auto', 'resource': 'b'}, "resource can only be 'n_samples' when max_resources='auto'"), ({'min_resources': 15, 'max_resources': 14}, 'min_resources_=15 is greater than max_resources_=14'), ({'cv': KFold(shuffle=True)}, 'must yield consistent folds'), ({'cv': ShuffleSplit()}, 'must yield consistent folds')])
def test_input_errors(Est, params, expected_error_message):
    base_estimator = FastClassifier()
    param_grid = {'a': [1]}
    X, y = make_classification(100)
    sh = Est(base_estimator, param_grid, **params)
    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)