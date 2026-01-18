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
@slow
def test_eigen_slow():
    from sympy.core.function import count_ops
    q = Symbol('q', positive=True)
    m = Matrix([[-2, exp(-q), 1], [exp(q), -2, 1], [1, 1, -2]])
    assert count_ops(m.eigenvals(simplify=False)) > count_ops(m.eigenvals(simplify=True))
    assert count_ops(m.eigenvals(simplify=lambda x: x)) > count_ops(m.eigenvals(simplify=True))