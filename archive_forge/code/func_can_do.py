from sympy.core.random import randrange
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, ON_CI, skip
from sympy.core.random import verify_numerically as tn
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
def can_do(ap, bq, numerical=True, div=1, lowerplane=False):
    r = hyperexpand(hyper(ap, bq, z))
    if r.has(hyper):
        return False
    if not numerical:
        return True
    repl = {}
    randsyms = r.free_symbols - {z}
    while randsyms:
        for n, ai in enumerate(randsyms):
            repl[ai] = randcplx(n) / div
        if not any((b.is_Integer and b <= 0 for b in Tuple(*bq).subs(repl))):
            break
    [a, b, c, d] = [2, -1, 3, 1]
    if lowerplane:
        [a, b, c, d] = [2, -2, 3, -1]
    return tn(hyper(ap, bq, z).subs(repl), r.replace(exp_polar, exp).subs(repl), z, a=a, b=b, c=c, d=d)