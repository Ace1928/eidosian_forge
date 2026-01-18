from __future__ import annotations
from typing import ClassVar, Iterator
from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
from .lie_group import _ode_lie_group
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
@cached_property
def eq_expanded(self) -> Expr:
    return expand(self.eq_preprocessed)