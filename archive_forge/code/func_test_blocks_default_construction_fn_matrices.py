import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_blocks_default_construction_fn_matrices():
    """Same idea as `test_default_construction_fn_matrices`, but for the block
    creation function"""
    A = scipy.sparse.coo_matrix(np.eye(2))
    B = scipy.sparse.coo_matrix([[2], [0]])
    C = scipy.sparse.coo_matrix([[3]])
    m = scipy.sparse.block_diag((A, B, C))
    assert not isinstance(m, scipy.sparse.sparray)
    m = scipy.sparse.bmat([[A, None], [None, C]])
    assert not isinstance(m, scipy.sparse.sparray)