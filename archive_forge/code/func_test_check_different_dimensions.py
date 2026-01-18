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
def test_check_different_dimensions():
    XA = np.resize(np.arange(45), (5, 9))
    XB = np.resize(np.arange(32), (4, 8))
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)
    XB = np.resize(np.arange(4 * 9), (4, 9))
    with pytest.raises(ValueError):
        check_paired_arrays(XA, XB)