import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_lstsqr(A, solver):
    x, *_ = getattr(spla, solver)(A, [1, 2, 3])
    npt.assert_allclose(A @ x, [1, 2, 3])