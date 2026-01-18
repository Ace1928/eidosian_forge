from functools import reduce
from sympy.core import Basic, S, Mul, PoleError, expand_mul
from sympy.core.cache import cacheit
from sympy.core.numbers import ilcm, I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up
from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis
@debug
@timeit
@cacheit
def mrv_leadterm(e, x):
    """Returns (c0, e0) for e."""
    Omega = SubsSet()
    if not e.has(x):
        return (e, S.Zero)
    if Omega == SubsSet():
        Omega, exps = mrv(e, x)
    if not Omega:
        return (exps, S.Zero)
    if x in Omega:
        Omega_up = moveup2(Omega, x)
        exps_up = moveup([exps], x)[0]
        Omega = Omega_up
        exps = exps_up
    w = Dummy('w', positive=True)
    f, logw = rewrite(exps, Omega, x, w)
    try:
        lt = f.leadterm(w, logx=logw)
    except (NotImplementedError, PoleError, ValueError):
        n0 = 1
        _series = Order(1)
        incr = S.One
        while _series.is_Order:
            _series = f._eval_nseries(w, n=n0 + incr, logx=logw)
            incr *= 2
        series = _series.expand().removeO()
        try:
            lt = series.leadterm(w, logx=logw)
        except (NotImplementedError, PoleError, ValueError):
            lt = f.as_coeff_exponent(w)
            if lt[0].has(w):
                base = f.as_base_exp()[0].as_coeff_exponent(w)
                ex = f.as_base_exp()[1]
                lt = (base[0] ** ex, base[1] * ex)
    return (lt[0].subs(log(w), logw), lt[1])