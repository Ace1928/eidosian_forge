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
def test_Type__cast_check__floating_point():
    raises(ValueError, lambda: f32.cast_check(123.45678949))
    raises(ValueError, lambda: f32.cast_check(12.345678949))
    raises(ValueError, lambda: f32.cast_check(1.2345678949))
    raises(ValueError, lambda: f32.cast_check(0.12345678949))
    assert abs(123.456789049 - f32.cast_check(123.456789049) - 4.9e-08) < 1e-08
    assert abs(0.12345678904 - f32.cast_check(0.12345678904) - 4e-11) < 1e-11
    dcm21 = Float('0.123456789012345670499')
    assert abs(dcm21 - f64.cast_check(dcm21) - 4.99e-19) < 1e-19
    f80.cast_check(Float('0.12345678901234567890103', precision=88))
    raises(ValueError, lambda: f80.cast_check(Float('0.12345678901234567890149', precision=88)))
    v10 = 12345.67894
    raises(ValueError, lambda: f32.cast_check(v10))
    assert abs(Float(str(v10), precision=64 + 8) - f64.cast_check(v10)) < v10 * 1e-16
    assert abs(f32.cast_check(2147483647) - 2147483650) < 1