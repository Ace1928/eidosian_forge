from collections import defaultdict
from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift
def collect_const(expr, *vars, Numbers=True):
    """A non-greedy collection of terms with similar number coefficients in
    an Add expr. If ``vars`` is given then only those constants will be
    targeted. Although any Number can also be targeted, if this is not
    desired set ``Numbers=False`` and no Float or Rational will be collected.

    Parameters
    ==========

    expr : SymPy expression
        This parameter defines the expression the expression from which
        terms with similar coefficients are to be collected. A non-Add
        expression is returned as it is.

    vars : variable length collection of Numbers, optional
        Specifies the constants to target for collection. Can be multiple in
        number.

    Numbers : bool
        Specifies to target all instance of
        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
        no Float or Rational will be collected.

    Returns
    =======

    expr : Expr
        Returns an expression with similar coefficient terms collected.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import s, x, y, z
    >>> from sympy.simplify.radsimp import collect_const
    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))
    sqrt(3)*(sqrt(2) + 2)
    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))
    (sqrt(3) + sqrt(7))*(s + 1)
    >>> s = sqrt(2) + 2
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))
    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))
    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)

    The collection is sign-sensitive, giving higher precedence to the
    unsigned values:

    >>> collect_const(x - y - z)
    x - (y + z)
    >>> collect_const(-y - z)
    -(y + z)
    >>> collect_const(2*x - 2*y - 2*z, 2)
    2*(x - y - z)
    >>> collect_const(2*x - 2*y - 2*z, -2)
    2*x - 2*(y + z)

    See Also
    ========

    collect, collect_sqrt, rcollect
    """
    if not expr.is_Add:
        return expr
    recurse = False
    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]
    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            q, r = f.div(Fv)
            if r.is_one:
                fwas = f.factors.copy()
                fnow = q.factors
                if not any((k in fwas and fwas[k].is_Integer and (not fnow[k].is_Integer) for k in fnow)):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)
        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue
            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k * v)
        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break
    return expr