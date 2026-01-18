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
@pytest.mark.parametrize('aggressive_elimination,max_resources,expected_n_iterations,expected_n_required_iterations,expected_n_possible_iterations,expected_n_remaining_candidates,expected_n_candidates,expected_n_resources,', [(True, 'limited', 4, 4, 3, 1, [60, 20, 7, 3], [20, 20, 60, 180]), (False, 'limited', 3, 4, 3, 3, [60, 20, 7], [20, 60, 180]), (True, 'unlimited', 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]), (False, 'unlimited', 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999])])
def test_aggressive_elimination(Est, aggressive_elimination, max_resources, expected_n_iterations, expected_n_required_iterations, expected_n_possible_iterations, expected_n_remaining_candidates, expected_n_candidates, expected_n_resources):
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': ('l1', 'l2'), 'b': list(range(30))}
    base_estimator = FastClassifier()
    if max_resources == 'limited':
        max_resources = 180
    else:
        max_resources = n_samples
    sh = Est(base_estimator, param_grid, aggressive_elimination=aggressive_elimination, max_resources=max_resources, factor=3)
    sh.set_params(verbose=True)
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=2 * 30, min_resources='exhaust')
    sh.fit(X, y)
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    assert sh.n_candidates_ == expected_n_candidates
    assert sh.n_remaining_candidates_ == expected_n_remaining_candidates
    assert ceil(sh.n_candidates_[-1] / sh.factor) == sh.n_remaining_candidates_