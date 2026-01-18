from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def eval_sign_at(p, vs):
    """
    Evaluate the sign of the polynomial `p` at `vs`.  `p` is a Z3
    Expression containing arithmetic operators: +, -, *, ^k where k is
    an integer; and free variables x that is_var(x) is True. Moreover,
    all variables must be real.

    The result is 1 if the polynomial is positive at the given point,
    -1 if negative, and 0 if zero.

    >>> x0, x1, x2 = RealVarVector(3)
    >>> eval_sign_at(x0**2 + x1*x2 + 1, (Numeral(0), Numeral(1), Numeral(2)))
    1
    >>> eval_sign_at(x0**2 - 2, [ Numeral(Sqrt(2)) ])
    0
    >>> eval_sign_at((x0 + x1)*(x0 + x2), (Numeral(0), Numeral(Sqrt(2)), Numeral(Sqrt(3))))
    1
    """
    num = len(vs)
    _vs = (Ast * num)()
    for i in range(num):
        _vs[i] = vs[i].ast
    return Z3_algebraic_eval(p.ctx_ref(), p.as_ast(), num, _vs)