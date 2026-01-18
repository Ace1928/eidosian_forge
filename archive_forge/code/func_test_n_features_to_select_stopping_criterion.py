import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('direction', ('forward', 'backward'))
def test_n_features_to_select_stopping_criterion(direction):
    """Check the behaviour stopping criterion for feature selection
    depending on the values of `n_features_to_select` and `tol`.

    When `direction` is `'forward'`, select a new features at random
    among those not currently selected in selector.support_,
    build a new version of the data that includes all the features
    in selector.support_ + this newly selected feature.
    And check that the cross-validation score of the model trained on
    this new dataset variant is lower than the model with
    the selected forward selected features or at least does not improve
    by more than the tol margin.

    When `direction` is `'backward'`, instead of adding a new feature
    to selector.support_, try to remove one of those selected features at random
    And check that the cross-validation score is either decreasing or
    not improving by more than the tol margin.
    """
    X, y = make_regression(n_features=50, n_informative=10, random_state=0)
    tol = 0.001
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select='auto', tol=tol, direction=direction, cv=2)
    sfs.fit(X, y)
    selected_X = sfs.transform(X)
    rng = np.random.RandomState(0)
    added_candidates = list(set(range(X.shape[1])) - set(sfs.get_support(indices=True)))
    added_X = np.hstack([selected_X, X[:, rng.choice(added_candidates)][:, np.newaxis]])
    removed_candidate = rng.choice(list(range(sfs.n_features_to_select_)))
    removed_X = np.delete(selected_X, removed_candidate, axis=1)
    plain_cv_score = cross_val_score(LinearRegression(), X, y, cv=2).mean()
    sfs_cv_score = cross_val_score(LinearRegression(), selected_X, y, cv=2).mean()
    added_cv_score = cross_val_score(LinearRegression(), added_X, y, cv=2).mean()
    removed_cv_score = cross_val_score(LinearRegression(), removed_X, y, cv=2).mean()
    assert sfs_cv_score >= plain_cv_score
    if direction == 'forward':
        assert sfs_cv_score - added_cv_score <= tol
        assert sfs_cv_score - removed_cv_score >= tol
    else:
        assert added_cv_score - sfs_cv_score <= tol
        assert removed_cv_score - sfs_cv_score <= tol