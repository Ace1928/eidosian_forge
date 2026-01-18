from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy
def test_Pauli():
    sigma1 = msigma(1)
    sigma2 = msigma(2)
    sigma3 = msigma(3)
    assert sigma1 == sigma1
    assert sigma1 != sigma2
    assert sigma1 * sigma2 == sigma3 * I
    assert sigma3 * sigma1 == sigma2 * I
    assert sigma2 * sigma3 == sigma1 * I
    assert sigma1 * sigma1 == eye(2)
    assert sigma2 * sigma2 == eye(2)
    assert sigma3 * sigma3 == eye(2)
    assert sigma1 * 2 * sigma1 == 2 * eye(2)
    assert sigma1 * sigma3 * sigma1 == -sigma3