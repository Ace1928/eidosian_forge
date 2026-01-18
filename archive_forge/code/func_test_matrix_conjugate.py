import pytest
import sys
from mpmath import *
def test_matrix_conjugate():
    A = matrix([[1 + j, 0], [2, j]])
    assert A.conjugate() == matrix([[mpc(1, -1), 0], [2, mpc(0, -1)]])
    assert A.transpose_conj() == A.H == matrix([[mpc(1, -1), 2], [0, mpc(0, -1)]])