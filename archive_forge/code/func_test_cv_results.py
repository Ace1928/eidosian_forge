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
@pytest.mark.parametrize('Est', (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_cv_results(Est):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': ('l1', 'l2'), 'b': list(range(30))}
    base_estimator = FastClassifier()

    def scorer(est, X, y):
        return rng.rand()
    sh = Est(base_estimator, param_grid, factor=2, scoring=scorer)
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=2 * 30, min_resources='exhaust')
    sh.fit(X, y)
    assert isinstance(sh.cv_results_['iter'], np.ndarray)
    assert isinstance(sh.cv_results_['n_resources'], np.ndarray)
    cv_results_df = pd.DataFrame(sh.cv_results_)
    assert len(cv_results_df['mean_test_score'].unique()) == len(cv_results_df)
    cv_results_df['params_str'] = cv_results_df['params'].apply(str)
    table = cv_results_df.pivot(index='params_str', columns='iter', values='mean_test_score')
    nan_mask = pd.isna(table)
    n_iter = sh.n_iterations_
    for it in range(n_iter - 1):
        already_discarded_mask = nan_mask[it]
        assert (already_discarded_mask & nan_mask[it + 1] == already_discarded_mask).all()
        discarded_now_mask = ~already_discarded_mask & nan_mask[it + 1]
        kept_mask = ~already_discarded_mask & ~discarded_now_mask
        assert kept_mask.sum() == sh.n_candidates_[it + 1]
        discarded_max_score = table[it].where(discarded_now_mask).max()
        kept_min_score = table[it].where(kept_mask).min()
        assert discarded_max_score < kept_min_score
    last_iter = cv_results_df['iter'].max()
    idx_best_last_iter = cv_results_df[cv_results_df['iter'] == last_iter]['mean_test_score'].idxmax()
    idx_best_all_iters = cv_results_df['mean_test_score'].idxmax()
    assert sh.best_params_ == cv_results_df.iloc[idx_best_last_iter]['params']
    assert cv_results_df.iloc[idx_best_last_iter]['mean_test_score'] < cv_results_df.iloc[idx_best_all_iters]['mean_test_score']
    assert cv_results_df.iloc[idx_best_last_iter]['params'] != cv_results_df.iloc[idx_best_all_iters]['params']