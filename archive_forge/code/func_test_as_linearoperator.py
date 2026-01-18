import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_as_linearoperator(A):
    L = spla.aslinearoperator(A)
    npt.assert_allclose(L * [1, 2, 3, 4], A @ [1, 2, 3, 4])