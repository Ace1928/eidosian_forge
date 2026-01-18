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
@pytest.mark.parametrize('SearchCV', [HalvingRandomSearchCV, HalvingGridSearchCV])
def test_min_resources_null(SearchCV):
    """Check that we raise an error if the minimum resources is set to 0."""
    base_estimator = FastClassifier()
    param_grid = {'a': [1]}
    X = np.empty(0).reshape(0, 3)
    search = SearchCV(base_estimator, param_grid, min_resources='smallest')
    err_msg = 'min_resources_=0: you might have passed an empty dataset X.'
    with pytest.raises(ValueError, match=err_msg):
        search.fit(X, [])