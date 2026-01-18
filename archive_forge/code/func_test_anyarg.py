from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)
from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
def test_anyarg():
    assert anyarg(x, Q.zero(x), x * y) == Or(Q.zero(x), Q.zero(y))
    assert anyarg(x, Q.positive(x) & Q.negative(x), x * y) == Or(Q.positive(x) & Q.negative(x), Q.positive(y) & Q.negative(y))