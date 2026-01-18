from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.logic.boolalg import And
from sympy.core.symbol import Str
from sympy.unify.core import Compound, Variable
from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
from sympy.abc import x, y, z, n
def test_Union():
    from sympy.sets.sets import Interval
    assert list(unify(Interval(0, 1) + Interval(10, 11), Interval(0, 1) + Interval(12, 13), variables=(Interval(12, 13),)))