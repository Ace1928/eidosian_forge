import pytest
import sys
from mpmath import *
@pytest.mark.xfail()
def test_interval_matrix_matrix_mult():
    """Multiplication of iv.matrix and other matrix types"""
    A = ones(1)
    B = fp.ones(1)
    M = iv.ones(1)
    for X in [A, B, M]:
        assert X * M == iv.matrix(X)
        assert X * M == X
        assert M * X == iv.matrix(X)
        assert M * X == X