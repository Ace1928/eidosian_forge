from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.matrices import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose
def test_definite():
    m = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[5, 4], [4, 5]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[1, 2], [2, 4]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[2, 3], [4, 8]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[1, 2 * I], [-I, 4]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    m = Matrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False
    m = Matrix([[b, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == True
    assert m.is_negative_semidefinite == True
    assert m.is_indefinite == False
    m = Matrix([[a, 0], [0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == True
    m = Matrix([[0.0228202735623867, 0.00518748979085398, -0.0743036351048907, -0.00709135324903921], [0.00518748979085398, 0.034904535978635, 0.0830317991056637, 0.00233147902806909], [-0.0743036351048907, 0.0830317991056637, 1.15859676366277, 0.340359081555988], [-0.00709135324903921, 0.00233147902806909, 0.340359081555988, 0.928147644848199]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_indefinite == False
    m = Matrix([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    assert not m.is_positive_definite
    assert not m.is_positive_semidefinite