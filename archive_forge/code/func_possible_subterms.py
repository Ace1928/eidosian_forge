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
def possible_subterms(term):
    if isinstance(term, (TrigonometricFunction, HyperbolicFunction, *inverse_trig_functions, exp, log, Heaviside)):
        return [term.args[0]]
    elif isinstance(term, (chebyshevt, chebyshevu, legendre, hermite, laguerre)):
        return [term.args[1]]
    elif isinstance(term, (gegenbauer, assoc_laguerre)):
        return [term.args[2]]
    elif isinstance(term, jacobi):
        return [term.args[3]]
    elif isinstance(term, Mul):
        r = []
        for u in term.args:
            r.append(u)
            r.extend(possible_subterms(u))
        return r
    elif isinstance(term, Pow):
        r = [arg for arg in term.args if arg.has(symbol)]
        if term.exp.is_Integer:
            r.extend([term.base ** d for d in primefactors(term.exp) if 1 < d < abs(term.args[1])])
            if term.base.is_Add:
                r.extend([t for t in possible_subterms(term.base) if t.is_Pow])
        return r
    elif isinstance(term, Add):
        r = []
        for arg in term.args:
            r.append(arg)
            r.extend(possible_subterms(arg))
        return r
    return []