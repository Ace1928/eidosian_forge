from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
def factor_nc(expr):
    """Return the factored form of ``expr`` while handling non-commutative
    expressions.

    Examples
    ========

    >>> from sympy import factor_nc, Symbol
    >>> from sympy.abc import x
    >>> A = Symbol('A', commutative=False)
    >>> B = Symbol('B', commutative=False)
    >>> factor_nc((x**2 + 2*A*x + A**2).expand())
    (x + A)**2
    >>> factor_nc(((x + A)*(x + B)).expand())
    (x + A)*(x + B)
    """
    expr = sympify(expr)
    if not isinstance(expr, Expr) or not expr.args:
        return expr
    if not expr.is_Add:
        return expr.func(*[factor_nc(a) for a in expr.args])
    expr = expr.func(*[expand_power_exp(i) for i in expr.args])
    from sympy.polys.polytools import gcd, factor
    expr, rep, nc_symbols = _mask_nc(expr)
    if rep:
        return factor(expr).subs(rep)
    else:
        args = [a.args_cnc() for a in Add.make_args(expr)]
        c = g = l = r = S.One
        hit = False
        for i, a in enumerate(args):
            if i == 0:
                c = Mul._from_args(a[0])
            elif a[0]:
                c = gcd(c, Mul._from_args(a[0]))
            else:
                c = S.One
        if c is not S.One:
            hit = True
            c, g = c.as_coeff_Mul()
            if g is not S.One:
                for i, (cc, _) in enumerate(args):
                    cc = list(Mul.make_args(Mul._from_args(list(cc)) / g))
                    args[i][0] = cc
            for i, (cc, _) in enumerate(args):
                if cc:
                    cc[0] = cc[0] / c
                else:
                    cc = [1 / c]
                args[i][0] = cc
        for i, a in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_prefix(n, a[1])
            if not n:
                if not args[0][1]:
                    break
                b, e = args[0][1][0].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        bt, et = t[1][0].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        l = b ** e
                        il = b ** (-e)
                        for _ in args:
                            _[1][0] = il * _[1][0]
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            l = Mul(*n)
            for _ in args:
                _[1] = _[1][lenn:]
        for i, a in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_suffix(n, a[1])
            if not n:
                if not args[0][1]:
                    break
                b, e = args[0][1][-1].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        bt, et = t[1][-1].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        r = b ** e
                        il = b ** (-e)
                        for _ in args:
                            _[1][-1] = _[1][-1] * il
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            r = Mul(*n)
            for _ in args:
                _[1] = _[1][:len(_[1]) - lenn]
        if hit:
            mid = Add(*[Mul(*cc) * Mul(*nc) for cc, nc in args])
        else:
            mid = expr
        from sympy.simplify.powsimp import powsimp
        rep1 = [(n, Dummy()) for n in sorted(nc_symbols, key=default_sort_key)]
        unrep1 = [(v, k) for k, v in rep1]
        unrep1.reverse()
        new_mid, r2, _ = _mask_nc(mid.subs(rep1))
        new_mid = powsimp(factor(new_mid))
        new_mid = new_mid.subs(r2).subs(unrep1)
        if new_mid.is_Pow:
            return _keep_coeff(c, g * l * new_mid * r)
        if new_mid.is_Mul:

            def _pemexpand(expr):
                """Expand with the minimal set of hints necessary to check the result."""
                return expr.expand(deep=True, mul=True, power_exp=True, power_base=False, basic=False, multinomial=True, log=False)
            cfac = []
            ncfac = []
            for f in new_mid.args:
                if f.is_commutative:
                    cfac.append(f)
                else:
                    b, e = f.as_base_exp()
                    if e.is_Integer:
                        ncfac.extend([b] * e)
                    else:
                        ncfac.append(f)
            pre_mid = g * Mul(*cfac) * l
            target = _pemexpand(expr / c)
            for s in variations(ncfac, len(ncfac)):
                ok = pre_mid * Mul(*s) * r
                if _pemexpand(ok) == target:
                    return _keep_coeff(c, ok)
        return _keep_coeff(c, g * l * mid * r)