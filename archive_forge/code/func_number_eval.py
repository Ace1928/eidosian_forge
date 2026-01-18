from .add import Add
from .exprtools import gcd_terms
from .function import Function
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .singleton import S
def number_eval(p, q):
    """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """
    if q.is_zero:
        raise ZeroDivisionError('Modulo by zero')
    if p is S.NaN or q is S.NaN or p.is_finite is False or (q.is_finite is False):
        return S.NaN
    if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
        return S.Zero
    if q.is_Number:
        if p.is_Number:
            return p % q
        if q == 2:
            if p.is_even:
                return S.Zero
            elif p.is_odd:
                return S.One
    if hasattr(p, '_eval_Mod'):
        rv = getattr(p, '_eval_Mod')(q)
        if rv is not None:
            return rv
    r = p / q
    if r.is_integer:
        return S.Zero
    try:
        d = int(r)
    except TypeError:
        pass
    else:
        if isinstance(d, int):
            rv = p - d * q
            if (rv * q < 0) == True:
                rv += q
            return rv
    d = abs(p)
    for _ in range(2):
        d -= abs(q)
        if d.is_negative:
            if q.is_positive:
                if p.is_positive:
                    return d + q
                elif p.is_negative:
                    return -d
            elif q.is_negative:
                if p.is_positive:
                    return d
                elif p.is_negative:
                    return -d + q
            break