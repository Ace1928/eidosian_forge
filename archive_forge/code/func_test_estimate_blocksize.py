from numpy import array, kron, diag
from numpy.testing import assert_, assert_equal
from scipy.sparse import _spfuncs as spfuncs
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from scipy.sparse._sparsetools import (csr_scale_rows, csr_scale_columns,
def test_estimate_blocksize(self):
    mats = []
    mats.append([[0, 1], [1, 0]])
    mats.append([[1, 1, 0], [0, 0, 1], [1, 0, 1]])
    mats.append([[0], [0], [1]])
    mats = [array(x) for x in mats]
    blks = []
    blks.append([[1]])
    blks.append([[1, 1], [1, 1]])
    blks.append([[1, 1], [0, 1]])
    blks.append([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
    blks = [array(x) for x in blks]
    for A in mats:
        for B in blks:
            X = kron(A, B)
            r, c = spfuncs.estimate_blocksize(X)
            assert_(r >= B.shape[0])
            assert_(c >= B.shape[1])