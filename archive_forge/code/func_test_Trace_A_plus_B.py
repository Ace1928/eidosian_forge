from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i
def test_Trace_A_plus_B():
    assert trace(A + B) == Trace(A) + Trace(B)
    assert Trace(A + B).arg == MatAdd(A, B)
    assert Trace(A + B).doit() == Trace(A) + Trace(B)