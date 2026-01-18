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
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.FitFailedWarning')
@pytest.mark.filterwarnings('ignore:Scoring failed:UserWarning')
@pytest.mark.filterwarnings('ignore:One or more of the:UserWarning')
@pytest.mark.parametrize('HalvingSearch', (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize('fail_at', ('fit', 'predict'))
def test_nan_handling(HalvingSearch, fail_at):
    """Check the selection of the best scores in presence of failure represented by
    NaN values."""
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    search = HalvingSearch(SometimesFailClassifier(), {f'fail_{fail_at}': [False, True], 'a': range(3)}, resource='n_estimators', max_resources=6, min_resources=1, factor=2)
    search.fit(X, y)
    assert not search.best_params_[f'fail_{fail_at}']
    scores = search.cv_results_['mean_test_score']
    ranks = search.cv_results_['rank_test_score']
    assert np.isnan(scores).any()
    unique_nan_ranks = np.unique(ranks[np.isnan(scores)])
    assert unique_nan_ranks.shape[0] == 1
    assert (unique_nan_ranks[0] >= ranks).all()