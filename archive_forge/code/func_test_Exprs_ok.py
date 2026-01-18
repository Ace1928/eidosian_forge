from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q
def test_Exprs_ok():
    rl = rewriterule(p + q, q + p, (p, q))
    next(rl(x + y)).is_commutative
    str(next(rl(x + y)))