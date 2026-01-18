from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_commutative():
    x, y = symbols('x y', commutative=False)
    assert dotprint(x + y) == dotprint(y + x)
    assert dotprint(x * y) != dotprint(y * x)