import pytest
import sys
from mpmath import *
def test_matrix_basic():
    A1 = matrix(3)
    for i in range(3):
        A1[i, i] = 1
    assert A1 == eye(3)
    assert A1 == matrix(A1)
    A2 = matrix(3, 2)
    assert not A2._matrix__data
    A3 = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert list(A3) == list(range(1, 10))
    A3[1, 1] = 0
    assert not (1, 1) in A3._matrix__data
    A4 = matrix([[1, 2, 3], [4, 5, 6]])
    A5 = matrix([[6, -1], [3, 2], [0, -3]])
    assert A4 * A5 == matrix([[12, -6], [39, -12]])
    assert A1 * A3 == A3 * A1 == A3
    pytest.raises(ValueError, lambda: A2 * A2)
    l = [[10, 20, 30], [40, 0, 60], [70, 80, 90]]
    A6 = matrix(l)
    assert A6.tolist() == l
    assert A6 == eval(repr(A6))
    A6 = fp.matrix(A6)
    assert A6 == eval(repr(A6))
    assert A6 * 1j == eval(repr(A6 * 1j))
    assert A3 * 10 == 10 * A3 == A6
    assert A2.rows == 3
    assert A2.cols == 2
    A3.rows = 2
    A3.cols = 2
    assert len(A3._matrix__data) == 3
    assert A4 + A4 == 2 * A4
    pytest.raises(ValueError, lambda: A4 + A2)
    assert sum(A1 - A1) == 0
    A7 = matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
    x = matrix([10, -10])
    assert A7 * x == matrix([-10, -10, -10, -10])
    A8 = ones(5)
    assert sum(A8 + 1 - (2 - zeros(5))) == 0
    assert (1 + ones(4)) / 2 - 1 == zeros(4)
    assert eye(3) ** 10 == eye(3)
    pytest.raises(ValueError, lambda: A7 ** 2)
    A9 = randmatrix(3)
    A10 = matrix(A9)
    A9[0, 0] = -100
    assert A9 != A10
    assert nstr(A9)