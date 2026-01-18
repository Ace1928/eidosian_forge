import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
def discriminant_func(sample, coef, intercept, clazz):
    return np.exp(intercept[clazz] + np.dot(sample, coef[clazz])).item()