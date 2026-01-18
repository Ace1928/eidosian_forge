from __future__ import annotations
import itertools
from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf
from sympy.utilities.timeutils import timethis
def meijerint_indefinite(f, x):
    """
    Compute an indefinite integral of ``f`` by rewriting it as a G function.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_indefinite
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> meijerint_indefinite(sin(x), x)
    -cos(x)
    """
    f = sympify(f)
    results = []
    for a in sorted(_find_splitting_points(f, x) | {S.Zero}, key=default_sort_key):
        res = _meijerint_indefinite_1(f.subs(x, x + a), x)
        if not res:
            continue
        res = res.subs(x, x - a)
        if _has(res, hyper, meijerg):
            results.append(res)
        else:
            return res
    if f.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_indefinite(_rewrite_hyperbolics_as_exp(f), x)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                return collect(factor_terms(rv), rv.atoms(exp))
            results.extend(rv)
    if results:
        return next(ordered(results))