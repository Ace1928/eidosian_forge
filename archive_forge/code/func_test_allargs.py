from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)
from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
def test_allargs():
    assert allargs(x, Q.zero(x), x * y) == And(Q.zero(x), Q.zero(y))
    assert allargs(x, Q.positive(x) | Q.negative(x), x * y) == And(Q.positive(x) | Q.negative(x), Q.positive(y) | Q.negative(y))