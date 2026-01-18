from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def analyse_gens(gens, hints):
    """
        Analyse the generators ``gens``, using the hints ``hints``.

        The meaning of ``hints`` is described in the main docstring.
        Return a new list of generators, and also the ideal we should
        work with.
        """
    n, funcs, iterables, extragens = parse_hints(hints)
    debug('n=%s   funcs: %s   iterables: %s    extragens: %s', (funcs, iterables, extragens))
    gens = list(gens)
    gens.extend(extragens)
    funcs = list(set(funcs))
    iterables = list(set(iterables))
    gens = list(set(gens))
    allfuncs = {sin, cos, tan, sinh, cosh, tanh}
    trigterms = [(g.args[0].as_coeff_mul(), g.func) for g in gens if g.func in allfuncs]
    freegens = [g for g in gens if g.func not in allfuncs]
    newgens = []
    trigdict = {}
    for (coeff, var), fn in trigterms:
        trigdict.setdefault(var, []).append((coeff, fn))
    res = []
    for key, val in trigdict.items():
        fns = [x[1] for x in val]
        val = [x[0] for x in val]
        gcd = reduce(igcd, val)
        terms = [(fn, v / gcd) for fn, v in zip(fns, val)]
        fs = set(funcs + fns)
        for c, s, t in ([cos, sin, tan], [cosh, sinh, tanh]):
            if any((x in fs for x in (c, s, t))):
                fs.add(c)
                fs.add(s)
        for fn in fs:
            for k in range(1, n + 1):
                terms.append((fn, k))
        extra = []
        for fn, v in terms:
            if fn == tan:
                extra.append((sin, v))
                extra.append((cos, v))
            if fn in [sin, cos] and tan in fs:
                extra.append((tan, v))
            if fn == tanh:
                extra.append((sinh, v))
                extra.append((cosh, v))
            if fn in [sinh, cosh] and tanh in fs:
                extra.append((tanh, v))
        terms.extend(extra)
        x = gcd * Mul(*key)
        r = build_ideal(x, terms)
        res.extend(r)
        newgens.extend({fn(v * x) for fn, v in terms})
    for fn, args in iterables:
        if fn == tan:
            iterables.extend([(sin, args), (cos, args)])
        elif fn == tanh:
            iterables.extend([(sinh, args), (cosh, args)])
        else:
            dummys = symbols('d:%i' % len(args), cls=Dummy)
            expr = fn(Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
            res.append(fn(Add(*args)) - expr)
    if myI in gens:
        res.append(myI ** 2 + 1)
        freegens.remove(myI)
        newgens.append(myI)
    return (res, freegens, newgens)