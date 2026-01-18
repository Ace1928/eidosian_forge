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
def test_gaussian_mixture_fit_predict_n_init():
    X = np.random.RandomState(0).randn(1000, 5)
    gm = GaussianMixture(n_components=5, n_init=5, random_state=0)
    y_pred1 = gm.fit_predict(X)
    y_pred2 = gm.predict(X)
    assert_array_equal(y_pred1, y_pred2)