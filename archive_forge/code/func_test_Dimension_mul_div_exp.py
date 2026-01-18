from sympy.physics.units.systems.si import dimsys_SI
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos)
from sympy.physics.units.dimensions import Dimension
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units import foot
from sympy.testing.pytest import raises
def test_Dimension_mul_div_exp():
    assert 2 * length == length * 2 == length / 2 == length
    assert 2 / length == 1 / length
    x = Symbol('x')
    m = x * length
    assert m == length * x and m.is_Mul and (set(m.args) == {x, length})
    d = x / length
    assert d == x * length ** (-1) and d.is_Mul and (set(d.args) == {x, 1 / length})
    d = length / x
    assert d == length * x ** (-1) and d.is_Mul and (set(d.args) == {1 / x, length})
    velo = length / time
    assert length * length == length ** 2
    assert dimsys_SI.get_dimensional_dependencies(length * length) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length ** 2) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length * time) == {length: 1, time: 1}
    assert dimsys_SI.get_dimensional_dependencies(velo) == {length: 1, time: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** 2) == {length: 2, time: -2}
    assert dimsys_SI.get_dimensional_dependencies(length / length) == {}
    assert dimsys_SI.get_dimensional_dependencies(velo / length * time) == {}
    assert dimsys_SI.get_dimensional_dependencies(length ** (-1)) == {length: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** (-1.5)) == {length: -1.5, time: 1.5}
    length_a = length ** 'a'
    assert dimsys_SI.get_dimensional_dependencies(length_a) == {length: Symbol('a')}
    assert dimsys_SI.get_dimensional_dependencies(length ** pi) == {length: pi}
    assert dimsys_SI.get_dimensional_dependencies(length ** (length / length)) == {length: Dimension(1)}
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(length ** length))
    assert length != 1
    assert length / length != 1
    length_0 = length ** 0
    assert dimsys_SI.get_dimensional_dependencies(length_0) == {}
    a = Symbol('a')
    b = Symbol('b')
    c = sqrt(a ** 2 + b ** 2)
    c_dim = c.subs({a: length, b: length})
    assert dimsys_SI.equivalent_dims(c_dim, length)