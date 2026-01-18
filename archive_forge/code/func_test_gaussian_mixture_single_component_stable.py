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
def test_gaussian_mixture_single_component_stable():
    """
    Non-regression test for #23032 ensuring 1-component GM works on only a
    few samples.
    """
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(np.zeros(2), np.identity(2), size=3)
    gm = GaussianMixture(n_components=1)
    gm.fit(X).sample()