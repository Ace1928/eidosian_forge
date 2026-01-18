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
def test_check_invalid_dimensions():
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)