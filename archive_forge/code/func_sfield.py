from __future__ import annotations
from typing import Any
from functools import reduce
from operator import add, mul, lt, le, gt, ge
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.constructor import construct_domain
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.rings import PolyElement
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
@public
def sfield(exprs, *symbols, **options):
    """Construct a field deriving generators and domain
    from options and input expressions.

    Parameters
    ==========

    exprs   : py:class:`~.Expr` or sequence of :py:class:`~.Expr` (sympifiable)

    symbols : sequence of :py:class:`~.Symbol`/:py:class:`~.Expr`

    options : keyword arguments understood by :py:class:`~.Options`

    Examples
    ========

    >>> from sympy import exp, log, symbols, sfield

    >>> x = symbols("x")
    >>> K, f = sfield((x*log(x) + 4*x**2)*exp(1/x + log(x)/3)/x**2)
    >>> K
    Rational function field in x, exp(1/x), log(x), x**(1/3) over ZZ with lex order
    >>> f
    (4*x**2*(exp(1/x)) + x*(exp(1/x))*(log(x)))/((x**(1/3))**5)
    """
    single = False
    if not is_sequence(exprs):
        exprs, single = ([exprs], True)
    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)
    numdens = []
    for expr in exprs:
        numdens.extend(expr.as_numer_denom())
    reps, opt = _parallel_dict_from_expr(numdens, opt)
    if opt.domain is None:
        coeffs = sum([list(rep.values()) for rep in reps], [])
        opt.domain, _ = construct_domain(coeffs, opt=opt)
    _field = FracField(opt.gens, opt.domain, opt.order)
    fracs = []
    for i in range(0, len(reps), 2):
        fracs.append(_field(tuple(reps[i:i + 2])))
    if single:
        return (_field, fracs[0])
    else:
        return (_field, fracs)