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
@pytest.mark.parametrize('min_resources,max_resources,expected_n_iterations,expected_n_possible_iterations,expected_n_resources,', [('smallest', 'auto', 2, 4, [20, 60]), (50, 'auto', 2, 3, [50, 150]), ('smallest', 30, 1, 1, [20]), ('exhaust', 'auto', 2, 2, [333, 999]), ('exhaust', 1000, 2, 2, [333, 999]), ('exhaust', 999, 2, 2, [333, 999]), ('exhaust', 600, 2, 2, [200, 600]), ('exhaust', 599, 2, 2, [199, 597]), ('exhaust', 300, 2, 2, [100, 300]), ('exhaust', 60, 2, 2, [20, 60]), ('exhaust', 50, 1, 1, [20]), ('exhaust', 20, 1, 1, [20])])
def test_min_max_resources(Est, min_resources, max_resources, expected_n_iterations, expected_n_possible_iterations, expected_n_resources):
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': [1, 2], 'b': [1, 2, 3]}
    base_estimator = FastClassifier()
    sh = Est(base_estimator, param_grid, factor=3, min_resources=min_resources, max_resources=max_resources)
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=6)
    sh.fit(X, y)
    expected_n_required_iterations = 2
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    if min_resources == 'exhaust':
        assert sh.n_possible_iterations_ == sh.n_iterations_ == len(sh.n_resources_)