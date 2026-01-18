from numpy import array, kron, diag
from numpy.testing import assert_, assert_equal
from scipy.sparse import _spfuncs as spfuncs
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from scipy.sparse._sparsetools import (csr_scale_rows, csr_scale_columns,
def test_count_blocks(self):

    def gold(A, bs):
        R, C = bs
        I, J = A.nonzero()
        return len(set(zip(I // R, J // C)))
    mats = []
    mats.append([[0]])
    mats.append([[1]])
    mats.append([[1, 0]])
    mats.append([[1, 1]])
    mats.append([[0, 1], [1, 0]])
    mats.append([[1, 1, 0], [0, 0, 1], [1, 0, 1]])
    mats.append([[0], [0], [1]])
    for A in mats:
        for B in mats:
            X = kron(A, B)
            Y = csr_matrix(X)
            for R in range(1, 6):
                for C in range(1, 6):
                    assert_equal(spfuncs.count_blocks(Y, (R, C)), gold(X, (R, C)))
    X = kron([[1, 1, 0], [0, 0, 1], [1, 0, 1]], [[1, 1]])
    Y = csc_matrix(X)
    assert_equal(spfuncs.count_blocks(X, (1, 2)), gold(X, (1, 2)))
    assert_equal(spfuncs.count_blocks(Y, (1, 2)), gold(X, (1, 2)))