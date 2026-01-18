from collections import defaultdict
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import (expand_log, count_ops, _mexpand,
from sympy.core.numbers import Float, I, pi, Rational
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int
import mpmath
def nc_simplify(expr, deep=True):
    """
    Simplify a non-commutative expression composed of multiplication
    and raising to a power by grouping repeated subterms into one power.
    Priority is given to simplifications that give the fewest number
    of arguments in the end (for example, in a*b*a*b*c*a*b*c simplifying
    to (a*b)**2*c*a*b*c gives 5 arguments while a*b*(a*b*c)**2 has 3).
    If ``expr`` is a sum of such terms, the sum of the simplified terms
    is returned.

    Keyword argument ``deep`` controls whether or not subexpressions
    nested deeper inside the main expression are simplified. See examples
    below. Setting `deep` to `False` can save time on nested expressions
    that do not need simplifying on all levels.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.simplify.simplify import nc_simplify
    >>> a, b, c = symbols("a b c", commutative=False)
    >>> nc_simplify(a*b*a*b*c*a*b*c)
    a*b*(a*b*c)**2
    >>> expr = a**2*b*a**4*b*a**4
    >>> nc_simplify(expr)
    a**2*(b*a**4)**2
    >>> nc_simplify(a*b*a*b*c**2*(a*b)**2*c**2)
    ((a*b)**2*c**2)**2
    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a)
    (a*b)**2 + 2*(a*c*a)**3
    >>> nc_simplify(b**-1*a**-1*(a*b)**2)
    a*b
    >>> nc_simplify(a**-1*b**-1*c*a)
    (b*a)**(-1)*c*a
    >>> expr = (a*b*a*b)**2*a*c*a*c
    >>> nc_simplify(expr)
    (a*b)**4*(a*c)**2
    >>> nc_simplify(expr, deep=False)
    (a*b*a*b)**2*(a*c)**2

    """
    if isinstance(expr, MatrixExpr):
        expr = expr.doit(inv_expand=False)
        _Add, _Mul, _Pow, _Symbol = (MatAdd, MatMul, MatPow, MatrixSymbol)
    else:
        _Add, _Mul, _Pow, _Symbol = (Add, Mul, Pow, Symbol)

    def _overlaps(args):
        m = [[[1, 0] if a == args[0] else [0] for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps = []
            j = 0
            for j in range(len(args) - i - 1):
                overlap = []
                for v in m[i - 1][j + 1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j + i + 1 + v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m

    def _reduce_inverses(_args):
        inv_tot = 0
        inverses = []
        args = []
        for arg in _args:
            if isinstance(arg, _Pow) and arg.args[1].is_extended_negative:
                inverses = [arg ** (-1)] + inverses
                inv_tot += 1
            else:
                if len(inverses) == 1:
                    args.append(inverses[0] ** (-1))
                elif len(inverses) > 1:
                    args.append(_Pow(_Mul(*inverses), -1))
                    inv_tot -= len(inverses) - 1
                inverses = []
                args.append(arg)
        if inverses:
            args.append(_Pow(_Mul(*inverses), -1))
            inv_tot -= len(inverses) - 1
        return (inv_tot, tuple(args))

    def get_score(s):
        if isinstance(s, _Pow):
            return get_score(s.args[0])
        elif isinstance(s, (_Add, _Mul)):
            return sum([get_score(a) for a in s.args])
        return 1

    def compare(s, alt_s):
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s
    if not isinstance(expr, (_Add, _Mul, _Pow)) or expr.is_commutative:
        return expr
    args = expr.args[:]
    if isinstance(expr, _Pow):
        if deep:
            return _Pow(nc_simplify(args[0]), args[1]).doit()
        else:
            return expr
    elif isinstance(expr, _Add):
        return _Add(*[nc_simplify(a, deep=deep) for a in args]).doit()
    else:
        c_args, args = expr.args_cnc()
        com_coeff = Mul(*c_args)
        if com_coeff != 1:
            return com_coeff * nc_simplify(expr / com_coeff, deep=deep)
    inv_tot, args = _reduce_inverses(args)
    invert = False
    if inv_tot > len(args) / 2:
        invert = True
        args = [a ** (-1) for a in args[::-1]]
    if deep:
        args = tuple((nc_simplify(a) for a in args))
    m = _overlaps(args)
    simps = {}
    post = 1
    pre = 1
    max_simp_coeff = 0
    simp = None
    for i in range(1, len(args)):
        simp_coeff = 0
        l = 0
        p = 0
        if i < len(args) - 1:
            rep = m[i][0]
        start = i
        end = i + 1
        if i == len(args) - 1 or rep == [0]:
            if isinstance(args[i], _Pow) and (not isinstance(args[i].args[0], _Symbol)):
                subterm = args[i].args[0].args
                l = len(subterm)
                if args[i - l:i] == subterm:
                    p += 1
                    start -= l
                if args[i + 1:i + 1 + l] == subterm:
                    p += 1
                    end += l
            if p:
                p += args[i].args[1]
            else:
                continue
        else:
            l = rep[0]
            start -= l - 1
            subterm = args[start:end]
            p = 2
            end += l
        if subterm in simps and simps[subterm] >= start:
            continue
        while end < len(args):
            if l in m[end - 1][0]:
                p += 1
                end += l
            elif isinstance(args[end], _Pow) and args[end].args[0].args == subterm:
                p += args[end].args[1]
                end += 1
            else:
                break
        pre_exp = 0
        pre_arg = 1
        if start - l >= 0 and args[start - l + 1:start] == subterm[1:]:
            if isinstance(subterm[0], _Pow):
                pre_arg = subterm[0].args[0]
                exp = subterm[0].args[1]
            else:
                pre_arg = subterm[0]
                exp = 1
            if isinstance(args[start - l], _Pow) and args[start - l].args[0] == pre_arg:
                pre_exp = args[start - l].args[1] - exp
                start -= l
                p += 1
            elif args[start - l] == pre_arg:
                pre_exp = 1 - exp
                start -= l
                p += 1
        post_exp = 0
        post_arg = 1
        if end + l - 1 < len(args) and args[end:end + l - 1] == subterm[:-1]:
            if isinstance(subterm[-1], _Pow):
                post_arg = subterm[-1].args[0]
                exp = subterm[-1].args[1]
            else:
                post_arg = subterm[-1]
                exp = 1
            if isinstance(args[end + l - 1], _Pow) and args[end + l - 1].args[0] == post_arg:
                post_exp = args[end + l - 1].args[1] - exp
                end += l
                p += 1
            elif args[end + l - 1] == post_arg:
                post_exp = 1 - exp
                end += l
                p += 1
        if post_exp and exp % 2 == 0 and (start > 0):
            exp = exp / 2
            _pre_exp = 1
            _post_exp = 1
            if isinstance(args[start - 1], _Pow) and args[start - 1].args[0] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = args[start - 1].args[1] - exp
            elif args[start - 1] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = 1 - exp
            if _pre_exp == 0 or _post_exp == 0:
                if not pre_exp:
                    start -= 1
                post_exp = _post_exp
                pre_exp = _pre_exp
                pre_arg = post_arg
                subterm = (post_arg ** exp,) + subterm[:-1] + (post_arg ** exp,)
        simp_coeff += end - start
        if post_exp:
            simp_coeff -= 1
        if pre_exp:
            simp_coeff -= 1
        simps[subterm] = end
        if simp_coeff > max_simp_coeff:
            max_simp_coeff = simp_coeff
            simp = (start, _Mul(*subterm), p, end, l)
            pre = pre_arg ** pre_exp
            post = post_arg ** post_exp
    if simp:
        subterm = _Pow(nc_simplify(simp[1], deep=deep), simp[2])
        pre = nc_simplify(_Mul(*args[:simp[0]]) * pre, deep=deep)
        post = post * nc_simplify(_Mul(*args[simp[3]:]), deep=deep)
        simp = pre * subterm * post
        if pre != 1 or post != 1:
            simp = nc_simplify(simp, deep=False)
    else:
        simp = _Mul(*args)
    if invert:
        simp = _Pow(simp, -1)
    if not isinstance(expr, MatrixExpr):
        f_expr = factor_nc(expr)
        if f_expr != expr:
            alt_simp = nc_simplify(f_expr, deep=deep)
            simp = compare(simp, alt_simp)
    else:
        simp = simp.doit(inv_expand=False)
    return simp