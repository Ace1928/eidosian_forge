from __future__ import annotations
from typing import NamedTuple, Type, Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Number, E
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Boolean
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
from sympy.functions.special.zeta_functions import polylog
from .integrals import Integral
from sympy.logic.boolalg import And
from sympy.ntheory.factor_ import primefactors
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.strategies.core import switch, do_one, null_safe, condition
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def integral_steps(integrand, symbol, **options):
    """Returns the steps needed to compute an integral.

    Explanation
    ===========

    This function attempts to mirror what a student would do by hand as
    closely as possible.

    SymPy Gamma uses this to provide a step-by-step explanation of an
    integral. The code it uses to format the results of this function can be
    found at
    https://github.com/sympy/sympy_gamma/blob/master/app/logic/intsteps.py.

    Examples
    ========

    >>> from sympy import exp, sin
    >>> from sympy.integrals.manualintegrate import integral_steps
    >>> from sympy.abc import x
    >>> print(repr(integral_steps(exp(x) / (1 + exp(2 * x)), x)))     # doctest: +NORMALIZE_WHITESPACE
    URule(integrand=exp(x)/(exp(2*x) + 1), variable=x, u_var=_u, u_func=exp(x),
    substep=ArctanRule(integrand=1/(_u**2 + 1), variable=_u, a=1, b=1, c=1))
    >>> print(repr(integral_steps(sin(x), x)))     # doctest: +NORMALIZE_WHITESPACE
    SinRule(integrand=sin(x), variable=x)
    >>> print(repr(integral_steps((x**2 + 3)**2, x)))     # doctest: +NORMALIZE_WHITESPACE
    RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
    substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
    substeps=[PowerRule(integrand=x**4, variable=x, base=x, exp=4),
    ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
    substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
    ConstantRule(integrand=9, variable=x)]))

    Returns
    =======

    rule : Rule
        The first step; most rules have substeps that must also be
        considered. These substeps can be evaluated using ``manualintegrate``
        to obtain a result.

    """
    cachekey = integrand.xreplace({symbol: _cache_dummy})
    if cachekey in _integral_cache:
        if _integral_cache[cachekey] is None:
            return DontKnowRule(integrand, symbol)
        else:
            return (_integral_cache[cachekey].xreplace(_cache_dummy, symbol), symbol)
    else:
        _integral_cache[cachekey] = None
    integral = IntegralInfo(integrand, symbol)

    def key(integral):
        integrand = integral.integrand
        if symbol not in integrand.free_symbols:
            return Number
        for cls in (Symbol, TrigonometricFunction, OrthogonalPolynomial):
            if isinstance(integrand, cls):
                return cls
        return type(integrand)

    def integral_is_subclass(*klasses):

        def _integral_is_subclass(integral):
            k = key(integral)
            return k and issubclass(k, klasses)
        return _integral_is_subclass
    result = do_one(null_safe(special_function_rule), null_safe(switch(key, {Pow: do_one(null_safe(power_rule), null_safe(inverse_trig_rule), null_safe(sqrt_linear_rule), null_safe(quadratic_denom_rule)), Symbol: power_rule, exp: exp_rule, Add: add_rule, Mul: do_one(null_safe(mul_rule), null_safe(trig_product_rule), null_safe(heaviside_rule), null_safe(quadratic_denom_rule), null_safe(sqrt_linear_rule), null_safe(sqrt_quadratic_rule)), Derivative: derivative_rule, TrigonometricFunction: trig_rule, Heaviside: heaviside_rule, DiracDelta: dirac_delta_rule, OrthogonalPolynomial: orthogonal_poly_rule, Number: constant_rule})), do_one(null_safe(trig_rule), null_safe(hyperbolic_rule), null_safe(alternatives(rewrites_rule, substitution_rule, condition(integral_is_subclass(Mul, Pow), partial_fractions_rule), condition(integral_is_subclass(Mul, Pow), cancel_rule), condition(integral_is_subclass(Mul, log, *inverse_trig_functions), parts_rule), condition(integral_is_subclass(Mul, Pow), distribute_expand_rule), trig_powers_products_rule, trig_expand_rule)), null_safe(condition(integral_is_subclass(Mul, Pow), nested_pow_rule)), null_safe(trig_substitution_rule)), fallback_rule)(integral)
    del _integral_cache[cachekey]
    return result