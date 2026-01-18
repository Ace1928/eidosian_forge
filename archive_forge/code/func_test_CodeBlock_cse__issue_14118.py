import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises
from sympy.codegen.ast import (
def test_CodeBlock_cse__issue_14118():
    c = CodeBlock(Assignment(A22, Matrix([[x, sin(y)], [3, 4]])), Assignment(B22, Matrix([[sin(y), 2 * sin(y)], [sin(y) ** 2, 7]])))
    assert c.cse() == CodeBlock(Assignment(x0, sin(y)), Assignment(A22, Matrix([[x, x0], [3, 4]])), Assignment(B22, Matrix([[x0, 2 * x0], [x0 ** 2, 7]])))