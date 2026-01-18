import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_min_max(A):
    if hasattr(A, 'min'):
        assert not isinstance(A.min(axis=1), np.matrix), 'Expected array, got matrix'
    if hasattr(A, 'max'):
        assert not isinstance(A.max(axis=1), np.matrix), 'Expected array, got matrix'
    if hasattr(A, 'argmin'):
        assert not isinstance(A.argmin(axis=1), np.matrix), 'Expected array, got matrix'
    if hasattr(A, 'argmax'):
        assert not isinstance(A.argmax(axis=1), np.matrix), 'Expected array, got matrix'