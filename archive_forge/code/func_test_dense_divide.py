import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_dense_divide(A):
    assert isinstance(A / 2, scipy.sparse.sparray), 'Expected array, got matrix'