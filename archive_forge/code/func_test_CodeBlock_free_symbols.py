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
def test_CodeBlock_free_symbols():
    c1 = CodeBlock(Assignment(x, y + z), Assignment(z, 1), Assignment(t, x), Assignment(y, 2))
    assert c1.free_symbols == set()
    c2 = CodeBlock(Assignment(x, y + z), Assignment(z, a * b), Assignment(t, x), Assignment(y, b + 3))
    assert c2.free_symbols == {a, b}