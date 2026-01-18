from operator import attrgetter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
def test_RFE_fit_score_params():

    class TestEstimator(BaseEstimator, ClassifierMixin):

        def fit(self, X, y, prop=None):
            if prop is None:
                raise ValueError('fit: prop cannot be None')
            self.svc_ = SVC(kernel='linear').fit(X, y)
            self.coef_ = self.svc_.coef_
            return self

        def score(self, X, y, prop=None):
            if prop is None:
                raise ValueError('score: prop cannot be None')
            return self.svc_.score(X, y)
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match='fit: prop cannot be None'):
        RFE(estimator=TestEstimator()).fit(X, y)
    with pytest.raises(ValueError, match='score: prop cannot be None'):
        RFE(estimator=TestEstimator()).fit(X, y, prop='foo').score(X, y)
    RFE(estimator=TestEstimator()).fit(X, y, prop='foo').score(X, y, prop='foo')