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
def test_FunctionCall():
    fc = FunctionCall('power', (x, 3))
    assert fc.function_args[0] == x
    assert fc.function_args[1] == 3
    assert len(fc.function_args) == 2
    assert isinstance(fc.function_args[1], Integer)
    assert fc == FunctionCall('power', (x, 3))
    assert fc != FunctionCall('power', (3, x))
    assert fc != FunctionCall('Power', (x, 3))
    assert fc.func(*fc.args) == fc
    fc2 = FunctionCall('fma', [2, 3, 4])
    assert len(fc2.function_args) == 3
    assert fc2.function_args[0] == 2
    assert fc2.function_args[1] == 3
    assert fc2.function_args[2] == 4
    assert str(fc2) in ('FunctionCall(fma, function_args=(2, 3, 4))', 'FunctionCall("fma", function_args=(2, 3, 4))')