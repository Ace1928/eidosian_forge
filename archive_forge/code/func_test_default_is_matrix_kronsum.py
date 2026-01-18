import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_default_is_matrix_kronsum():
    m = scipy.sparse.kronsum(np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]))
    assert not isinstance(m, scipy.sparse.sparray)