import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_dense_addition(A):
    X = np.random.random(A.shape)
    assert not isinstance(A + X, np.matrix), 'Expected array, got matrix'