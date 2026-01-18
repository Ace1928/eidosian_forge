from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy
def test_Dirac():
    gamma0 = mgamma(0)
    gamma1 = mgamma(1)
    gamma2 = mgamma(2)
    gamma3 = mgamma(3)
    gamma5 = mgamma(5)
    assert gamma5 == gamma0 * gamma1 * gamma2 * gamma3 * I
    assert gamma1 * gamma2 + gamma2 * gamma1 == zeros(4)
    assert gamma0 * gamma0 == eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 != eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 == eye(4) * minkowski_tensor[2, 2]
    assert mgamma(5, True) == mgamma(0, True) * mgamma(1, True) * mgamma(2, True) * mgamma(3, True) * I