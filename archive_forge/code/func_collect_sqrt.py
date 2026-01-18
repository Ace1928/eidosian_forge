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
def collect_sqrt(expr, evaluate=None):
    """Return expr with terms having common square roots collected together.
    If ``evaluate`` is False a count indicating the number of sqrt-containing
    terms will be returned and, if non-zero, the terms of the Add will be
    returned, else the expression itself will be returned as a single term.
    If ``evaluate`` is True, the expression with any collected terms will be
    returned.

    Note: since I = sqrt(-1), it is collected, too.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b

    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]
    >>> collect_sqrt(a*r2 + b*r2)
    sqrt(2)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)
    sqrt(2)*(a + b) + sqrt(3)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)
    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)

    If evaluate is False then the arguments will be sorted and
    returned as a list and a count of the number of sqrt-containing
    terms will be returned:

    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)
    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)
    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)
    ((b, sqrt(2)*a), 1)
    >>> collect_sqrt(a + b, evaluate=False)
    ((a + b,), 0)

    See Also
    ========

    collect, collect_const, rcollect
    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    coeff, expr = expr.as_content_primitive()
    vars = set()
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (m.is_Pow and m.exp.is_Rational and (m.exp.q == 2) or m is S.ImaginaryUnit):
                vars.add(m)
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d
    if not evaluate:
        nrad = 0
        args = list(ordered(Add.make_args(d)))
        for i, m in enumerate(args):
            c, nc = m.args_cnc()
            for ci in c:
                if ci.is_Pow and ci.exp.is_Rational and (ci.exp.q == 2) or ci is S.ImaginaryUnit:
                    nrad += 1
                    break
            args[i] *= coeff
        if not (hit or nrad):
            args = [Add(*args)]
        return (tuple(args), nrad)
    return coeff * d