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
def collect_abs(expr):
    """Return ``expr`` with arguments of multiple Abs in a term collected
    under a single instance.

    Examples
    ========

    >>> from sympy.simplify.radsimp import collect_abs
    >>> from sympy.abc import x
    >>> collect_abs(abs(x + 1)/abs(x**2 - 1))
    Abs((x + 1)/(x**2 - 1))
    >>> collect_abs(abs(1/x))
    Abs(1/x)
    """

    def _abs(mul):
        c, nc = mul.args_cnc()
        a = []
        o = []
        for i in c:
            if isinstance(i, Abs):
                a.append(i.args[0])
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
                a.append(i.base.args[0] ** i.exp)
            else:
                o.append(i)
        if len(a) < 2 and (not any((i.exp.is_negative for i in a if isinstance(i, Pow)))):
            return mul
        absarg = Mul(*a)
        A = Abs(absarg)
        args = [A]
        args.extend(o)
        if not A.has(Abs):
            args.extend(nc)
            return Mul(*args)
        if not isinstance(A, Abs):
            A = Abs(absarg, evaluate=False)
        args[0] = A
        _mulsort(args)
        args.extend(nc)
        return Mul._from_args(args, is_commutative=not nc)
    return expr.replace(lambda x: isinstance(x, Mul), lambda x: _abs(x)).replace(lambda x: isinstance(x, Pow), lambda x: _abs(x))