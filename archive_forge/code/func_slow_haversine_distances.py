import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
def slow_haversine_distances(x, y):
    diff_lat = y[0] - x[0]
    diff_lon = y[1] - x[1]
    a = np.sin(diff_lat / 2) ** 2 + np.cos(x[0]) * np.cos(y[0]) * np.sin(diff_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c