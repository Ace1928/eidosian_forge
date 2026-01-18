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
def special_function_rule(integral):
    integrand, symbol = integral
    if not _special_function_patterns:
        a = Wild('a', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        b = Wild('b', exclude=[_symbol])
        c = Wild('c', exclude=[_symbol])
        d = Wild('d', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        e = Wild('e', exclude=[_symbol], properties=[lambda x: not (x.is_nonnegative and x.is_integer)])
        _wilds.extend((a, b, c, d, e))
        linear_pattern = a * _symbol + b
        quadratic_pattern = a * _symbol ** 2 + b * _symbol + c
        _special_function_patterns.extend(((Mul, exp(linear_pattern, evaluate=False) / _symbol, None, EiRule), (Mul, cos(linear_pattern, evaluate=False) / _symbol, None, CiRule), (Mul, cosh(linear_pattern, evaluate=False) / _symbol, None, ChiRule), (Mul, sin(linear_pattern, evaluate=False) / _symbol, None, SiRule), (Mul, sinh(linear_pattern, evaluate=False) / _symbol, None, ShiRule), (Pow, 1 / log(linear_pattern, evaluate=False), None, LiRule), (exp, exp(quadratic_pattern, evaluate=False), None, ErfRule), (sin, sin(quadratic_pattern, evaluate=False), None, FresnelSRule), (cos, cos(quadratic_pattern, evaluate=False), None, FresnelCRule), (Mul, _symbol ** e * exp(a * _symbol, evaluate=False), None, UpperGammaRule), (Mul, polylog(b, a * _symbol, evaluate=False) / _symbol, None, PolylogRule), (Pow, 1 / sqrt(a - d * sin(_symbol, evaluate=False) ** 2), lambda a, d: a != d, EllipticFRule), (Pow, sqrt(a - d * sin(_symbol, evaluate=False) ** 2), lambda a, d: a != d, EllipticERule)))
    _integrand = integrand.subs(symbol, _symbol)
    for type_, pattern, constraint, rule in _special_function_patterns:
        if isinstance(_integrand, type_):
            match = _integrand.match(pattern)
            if match:
                wild_vals = tuple((match.get(w) for w in _wilds if match.get(w) is not None))
                if constraint is None or constraint(*wild_vals):
                    return rule(integrand, symbol, *wild_vals)