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
def test_Attribute():
    noexcept = Attribute('noexcept')
    assert noexcept == Attribute('noexcept')
    alignas16 = Attribute('alignas', [16])
    alignas32 = Attribute('alignas', [32])
    assert alignas16 != alignas32
    assert alignas16.func(*alignas16.args) == alignas16