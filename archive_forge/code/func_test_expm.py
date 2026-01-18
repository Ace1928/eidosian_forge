import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_square_sparrays
def test_expm(B):
    if B.__class__.__name__[:3] != 'csc':
        return
    Bmat = scipy.sparse.csc_matrix(B)
    C = spla.expm(B)
    assert isinstance(C, scipy.sparse.sparray)
    npt.assert_allclose(C.todense(), spla.expm(Bmat).todense())