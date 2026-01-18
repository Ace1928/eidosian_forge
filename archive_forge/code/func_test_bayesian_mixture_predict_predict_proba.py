import copy
import numpy as np
import pytest
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm, _log_wishart_norm
from sklearn.mixture.tests.test_gaussian_mixture import RandomData
from sklearn.utils._testing import (
def test_bayesian_mixture_predict_predict_proba():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    for prior_type in PRIOR_TYPE:
        for covar_type in COVARIANCE_TYPE:
            X = rand_data.X[covar_type]
            Y = rand_data.Y
            bgmm = BayesianGaussianMixture(n_components=rand_data.n_components, random_state=rng, weight_concentration_prior_type=prior_type, covariance_type=covar_type)
            msg = "This BayesianGaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            with pytest.raises(NotFittedError, match=msg):
                bgmm.predict(X)
            bgmm.fit(X)
            Y_pred = bgmm.predict(X)
            Y_pred_proba = bgmm.predict_proba(X).argmax(axis=1)
            assert_array_equal(Y_pred, Y_pred_proba)
            assert adjusted_rand_score(Y, Y_pred) >= 0.95