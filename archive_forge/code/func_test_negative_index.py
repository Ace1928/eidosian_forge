from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions import MatrixSymbol
from sympy.abc import a, b, c, d, k, l, m, n
from sympy.testing.pytest import raises, XFAIL
from sympy.functions.elementary.integers import floor
from sympy.assumptions import assuming, Q
def test_negative_index():
    X = MatrixSymbol('x', 10, 10)
    assert X[-1, :] == X[9, :]