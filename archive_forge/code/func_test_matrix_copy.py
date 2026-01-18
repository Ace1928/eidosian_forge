import pytest
import sys
from mpmath import *
def test_matrix_copy():
    A = ones(6)
    B = A.copy()
    C = +A
    assert A == B
    assert A == C
    B[0, 0] = 0
    assert A != B
    C[0, 0] = 42
    assert A != C