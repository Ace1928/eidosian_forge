from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve
from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
from sympy.integrals.meijerint import _mytype
def from_hyper(func, x0=0, evalf=False):
    """
    Converts a hypergeometric function to holonomic.
    ``func`` is the Hypergeometric Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_hyper
    >>> from sympy import symbols, hyper, S
    >>> x = symbols('x')
    >>> from_hyper(hyper([], [S(3)/2], x**2/4))
    HolonomicFunction((-x) + (2)*Dx + (x)*Dx**2, x, 1, [sinh(1), -sinh(1) + cosh(1)])
    """
    a = func.ap
    b = func.bq
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    xDx = x * Dx
    r1 = 1
    for ai in a:
        r1 *= xDx + ai
    xDx_1 = xDx - 1
    r2 = Dx
    for bi in b:
        r2 *= xDx_1 + bi
    sol = r1 - r2
    simp = hyperexpand(func)
    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    def _find_conditions(simp, x, x0, order, evalf=False):
        y0 = []
        for i in range(order):
            if evalf:
                val = simp.subs(x, x0).evalf()
            else:
                val = simp.subs(x, x0)
            if val.is_finite is False or isinstance(val, NaN):
                return None
            y0.append(val)
            simp = simp.diff(x)
        return y0
    if not isinstance(simp, hyper):
        y0 = _find_conditions(simp, x, x0, sol.order)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order)
        return HolonomicFunction(sol, x).composition(z, x0, y0)
    if isinstance(simp, hyper):
        x0 = 1
        y0 = _find_conditions(simp, x, x0, sol.order, evalf)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf)
        return HolonomicFunction(sol, x).composition(z, x0, y0)
    return HolonomicFunction(sol, x).composition(z)