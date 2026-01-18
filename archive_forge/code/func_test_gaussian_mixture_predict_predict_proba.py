import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_gaussian_mixture_predict_predict_proba():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = GaussianMixture(n_components=rand_data.n_components, random_state=rng, weights_init=rand_data.weights, means_init=rand_data.means, precisions_init=rand_data.precisions[covar_type], covariance_type=covar_type)
        msg = "This GaussianMixture instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        with pytest.raises(NotFittedError, match=msg):
            g.predict(X)
        g.fit(X)
        Y_pred = g.predict(X)
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        assert_array_equal(Y_pred, Y_pred_proba)
        assert adjusted_rand_score(Y, Y_pred) > 0.95