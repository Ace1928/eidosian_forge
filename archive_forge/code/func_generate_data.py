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
def generate_data(n_samples, n_features, weights, means, precisions, covariance_type):
    rng = np.random.RandomState(0)
    X = []
    if covariance_type == 'spherical':
        for _, (w, m, c) in enumerate(zip(weights, means, precisions['spherical'])):
            X.append(rng.multivariate_normal(m, c * np.eye(n_features), int(np.round(w * n_samples))))
    if covariance_type == 'diag':
        for _, (w, m, c) in enumerate(zip(weights, means, precisions['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c), int(np.round(w * n_samples))))
    if covariance_type == 'tied':
        for _, (w, m) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, precisions['tied'], int(np.round(w * n_samples))))
    if covariance_type == 'full':
        for _, (w, m, c) in enumerate(zip(weights, means, precisions['full'])):
            X.append(rng.multivariate_normal(m, c, int(np.round(w * n_samples))))
    X = np.vstack(X)
    return X