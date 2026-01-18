from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def isolate_roots(p, vs=[]):
    """
    Given a multivariate polynomial p(x_0, ..., x_{n-1}, x_n), returns the
    roots of the univariate polynomial p(vs[0], ..., vs[len(vs)-1], x_n).

    Remarks:
    * p is a Z3 expression that contains only arithmetic terms and free variables.
    * forall i in [0, n) vs is a numeral.

    The result is a list of numerals

    >>> x0 = RealVar(0)
    >>> isolate_roots(x0**5 - x0 - 1)
    [1.1673039782?]
    >>> x1 = RealVar(1)
    >>> isolate_roots(x0**2 - x1**4 - 1, [ Numeral(Sqrt(3)) ])
    [-1.1892071150?, 1.1892071150?]
    >>> x2 = RealVar(2)
    >>> isolate_roots(x2**2 + x0 - x1, [ Numeral(Sqrt(3)), Numeral(Sqrt(2)) ])
    []
    """
    num = len(vs)
    _vs = (Ast * num)()
    for i in range(num):
        _vs[i] = vs[i].ast
    _roots = AstVector(Z3_algebraic_roots(p.ctx_ref(), p.as_ast(), num, _vs), p.ctx)
    return [Numeral(r) for r in _roots]