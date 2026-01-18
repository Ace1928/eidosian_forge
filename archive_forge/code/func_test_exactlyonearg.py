from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)
from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
def test_exactlyonearg():
    assert exactlyonearg(x, Q.zero(x), x * y) == Or(Q.zero(x) & ~Q.zero(y), Q.zero(y) & ~Q.zero(x))
    assert exactlyonearg(x, Q.zero(x), x * y * z) == Or(Q.zero(x) & ~Q.zero(y) & ~Q.zero(z), Q.zero(y) & ~Q.zero(x) & ~Q.zero(z), Q.zero(z) & ~Q.zero(x) & ~Q.zero(y))
    assert exactlyonearg(x, Q.positive(x) | Q.negative(x), x * y) == Or((Q.positive(x) | Q.negative(x)) & ~(Q.positive(y) | Q.negative(y)), (Q.positive(y) | Q.negative(y)) & ~(Q.positive(x) | Q.negative(x)))