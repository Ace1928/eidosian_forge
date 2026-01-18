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
def test_hard_match():
    from sympy.functions.elementary.trigonometric import cos, sin
    expr = sin(x) + cos(x) ** 2
    p, q = map(Symbol, 'pq')
    pattern = sin(p) + cos(p) ** 2
    assert list(unify(expr, pattern, {}, (p, q))) == [{p: x}]