from .products import product
from .summations import Sum, summation
from sympy.core import Add, Mul, S, Dummy
from sympy.core.cache import cacheit
from sympy.core.sorting import default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold
from sympy.polys.polytools import factor
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
@cacheit
def deltaproduct(f, limit):
    """
    Handle products containing a KroneckerDelta.

    See Also
    ========

    deltasummation
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.products.product
    """
    if (limit[2] - limit[1] < 0) == True:
        return S.One
    if not f.has(KroneckerDelta):
        return product(f, limit)
    if f.is_Add:
        delta = None
        terms = []
        for arg in sorted(f.args, key=default_sort_key):
            if delta is None and _has_simple_delta(arg, limit[0]):
                delta = arg
            else:
                terms.append(arg)
        newexpr = f.func(*terms)
        k = Dummy('kprime', integer=True)
        if isinstance(limit[1], int) and isinstance(limit[2], int):
            result = deltaproduct(newexpr, limit) + sum([deltaproduct(newexpr, (limit[0], limit[1], ik - 1)) * delta.subs(limit[0], ik) * deltaproduct(newexpr, (limit[0], ik + 1, limit[2])) for ik in range(int(limit[1]), int(limit[2] + 1))])
        else:
            result = deltaproduct(newexpr, limit) + deltasummation(deltaproduct(newexpr, (limit[0], limit[1], k - 1)) * delta.subs(limit[0], k) * deltaproduct(newexpr, (limit[0], k + 1, limit[2])), (k, limit[1], limit[2]), no_piecewise=_has_simple_delta(newexpr, limit[0]))
        return _remove_multiple_delta(result)
    delta, _ = _extract_delta(f, limit[0])
    if not delta:
        g = _expand_delta(f, limit[0])
        if f != g:
            try:
                return factor(deltaproduct(g, limit))
            except AssertionError:
                return deltaproduct(g, limit)
        return product(f, limit)
    return _remove_multiple_delta(f.subs(limit[0], limit[1]) * KroneckerDelta(limit[2], limit[1])) + S.One * _simplify_delta(KroneckerDelta(limit[2], limit[1] - 1))