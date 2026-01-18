import pytest
import sys
from mpmath import *
def test_matrix_numpy():
    try:
        import numpy
    except ImportError:
        return
    l = [[1, 2], [3, 4], [5, 6]]
    a = numpy.array(l)
    assert matrix(l) == matrix(a)