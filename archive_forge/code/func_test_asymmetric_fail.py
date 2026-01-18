import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg._isolve import minres
from pytest import raises as assert_raises
def test_asymmetric_fail():
    """Asymmetric matrix should raise `ValueError` when check=True"""
    A, b = get_sample_problem()
    A[1, 2] = 1
    A[2, 1] = 2
    with assert_raises(ValueError):
        xp, info = minres(A, b, check=True)