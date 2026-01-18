from sympy.matrices.sparsetools import _doktocsr, _csrtodok, banded
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices import SparseMatrix
from sympy.testing.pytest import raises
def test_csrtodok():
    h = [[5, 7, 5], [2, 1, 3], [0, 1, 1, 3], [3, 4]]
    g = [[12, 5, 4], [2, 4, 2], [0, 1, 2, 3], [3, 7]]
    i = [[1, 3, 12], [0, 2, 4], [0, 2, 3], [2, 5]]
    j = [[11, 15, 12, 15], [2, 4, 1, 2], [0, 1, 1, 2, 3, 4], [5, 8]]
    k = [[1, 3], [2, 1], [0, 1, 1, 2], [3, 3]]
    m = _csrtodok(h)
    assert isinstance(m, SparseMatrix)
    assert m == SparseMatrix(3, 4, {(0, 2): 5, (2, 1): 7, (2, 3): 5})
    assert _csrtodok(g) == SparseMatrix(3, 7, {(0, 2): 12, (1, 4): 5, (2, 2): 4})
    assert _csrtodok(i) == SparseMatrix([[1, 0, 3, 0, 0], [0, 0, 0, 0, 12]])
    assert _csrtodok(j) == SparseMatrix(5, 8, {(0, 2): 11, (2, 4): 15, (3, 1): 12, (4, 2): 15})
    assert _csrtodok(k) == SparseMatrix(3, 3, {(0, 2): 1, (2, 1): 3})