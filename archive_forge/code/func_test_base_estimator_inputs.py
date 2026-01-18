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
def test_base_estimator_inputs(Est):
    pd = pytest.importorskip('pandas')
    passed_n_samples_fit = []
    passed_n_samples_predict = []
    passed_params = []

    class FastClassifierBookKeeping(FastClassifier):

        def fit(self, X, y):
            passed_n_samples_fit.append(X.shape[0])
            return super().fit(X, y)

        def predict(self, X):
            passed_n_samples_predict.append(X.shape[0])
            return super().predict(X)

        def set_params(self, **params):
            passed_params.append(params)
            return super().set_params(**params)
    n_samples = 1024
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': ('l1', 'l2'), 'b': list(range(30))}
    base_estimator = FastClassifierBookKeeping()
    sh = Est(base_estimator, param_grid, factor=2, cv=n_splits, return_train_score=False, refit=False)
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=2 * 30, min_resources='exhaust')
    sh.fit(X, y)
    assert len(passed_n_samples_fit) == len(passed_n_samples_predict)
    passed_n_samples = [x + y for x, y in zip(passed_n_samples_fit, passed_n_samples_predict)]
    passed_n_samples = passed_n_samples[::n_splits]
    passed_params = passed_params[::n_splits]
    cv_results_df = pd.DataFrame(sh.cv_results_)
    assert len(passed_params) == len(passed_n_samples) == len(cv_results_df)
    uniques, counts = np.unique(passed_n_samples, return_counts=True)
    assert (sh.n_resources_ == uniques).all()
    assert (sh.n_candidates_ == counts).all()
    assert (cv_results_df['params'] == passed_params).all()
    assert (cv_results_df['n_resources'] == passed_n_samples).all()