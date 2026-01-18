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
def expr_to_holonomic(func, x=None, x0=0, y0=None, lenics=None, domain=None, initcond=True):
    """
    Converts a function or an expression to a holonomic function.

    Parameters
    ==========

    func:
        The expression to be converted.
    x:
        variable for the function.
    x0:
        point at which initial condition must be computed.
    y0:
        One can optionally provide initial condition if the method
        is not able to do it automatically.
    lenics:
        Number of terms in the initial condition. By default it is
        equal to the order of the annihilator.
    domain:
        Ground domain for the polynomials in ``x`` appearing as coefficients
        in the annihilator.
    initcond:
        Set it false if you do not want the initial conditions to be computed.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import expr_to_holonomic
    >>> from sympy import sin, exp, symbols
    >>> x = symbols('x')
    >>> expr_to_holonomic(sin(x))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1])
    >>> expr_to_holonomic(exp(x))
    HolonomicFunction((-1) + (1)*Dx, x, 0, [1])

    See Also
    ========

    sympy.integrals.meijerint._rewrite1, _convert_poly_rat_alg, _create_table
    """
    func = sympify(func)
    syms = func.free_symbols
    if not x:
        if len(syms) == 1:
            x = syms.pop()
        else:
            raise ValueError('Specify the variable for the function')
    elif x in syms:
        syms.remove(x)
    extra_syms = list(syms)
    if domain is None:
        if func.has(Float):
            domain = RR
        else:
            domain = QQ
        if len(extra_syms) != 0:
            domain = domain[extra_syms].get_field()
    solpoly = _convert_poly_rat_alg(func, x, x0=x0, y0=y0, lenics=lenics, domain=domain, initcond=initcond)
    if solpoly:
        return solpoly
    global _lookup_table, domain_for_table
    if not _lookup_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)
    elif domain != domain_for_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)
    if func.is_Function:
        f = func.subs(x, x_1)
        t = _mytype(f, x_1)
        if t in _lookup_table:
            l = _lookup_table[t]
            sol = l[0][1].change_x(x)
        else:
            sol = _convert_meijerint(func, x, initcond=False, domain=domain)
            if not sol:
                raise NotImplementedError
            if y0:
                sol.y0 = y0
            if y0 or not initcond:
                sol.x0 = x0
                return sol
            if not lenics:
                lenics = sol.annihilator.order
            _y0 = _find_conditions(func, x, x0, lenics)
            while not _y0:
                x0 += 1
                _y0 = _find_conditions(func, x, x0, lenics)
            return HolonomicFunction(sol.annihilator, x, x0, _y0)
        if y0 or not initcond:
            sol = sol.composition(func.args[0])
            if y0:
                sol.y0 = y0
            sol.x0 = x0
            return sol
        if not lenics:
            lenics = sol.annihilator.order
        _y0 = _find_conditions(func, x, x0, lenics)
        while not _y0:
            x0 += 1
            _y0 = _find_conditions(func, x, x0, lenics)
        return sol.composition(func.args[0], x0, _y0)
    args = func.args
    f = func.func
    sol = expr_to_holonomic(args[0], x=x, initcond=False, domain=domain)
    if f is Add:
        for i in range(1, len(args)):
            sol += expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)
    elif f is Mul:
        for i in range(1, len(args)):
            sol *= expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)
    elif f is Pow:
        sol = sol ** args[1]
    sol.x0 = x0
    if not sol:
        raise NotImplementedError
    if y0:
        sol.y0 = y0
    if y0 or not initcond:
        return sol
    if sol.y0:
        return sol
    if not lenics:
        lenics = sol.annihilator.order
    if sol.annihilator.is_singular(x0):
        r = sol._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0) ** r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r: singular_ics}
            return HolonomicFunction(sol.annihilator, x, x0, y0)
    _y0 = _find_conditions(func, x, x0, lenics)
    while not _y0:
        x0 += 1
        _y0 = _find_conditions(func, x, x0, lenics)
    return HolonomicFunction(sol.annihilator, x, x0, _y0)