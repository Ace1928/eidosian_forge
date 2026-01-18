import re
import warnings
from enum import Enum
from math import gcd
def as_term_coeff(obj):
    """Return expression as term-coefficient pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.INTEGER:
            return (as_integer(1, obj.data[1]), obj.data[0])
        if obj.op is Op.REAL:
            return (as_real(1, obj.data[1]), obj.data[0])
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                return (term, coeff)
        if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
            t, c = as_term_coeff(obj.data[1][0])
            return (as_apply(ArithOp.DIV, t, obj.data[1][1]), c)
        return (obj, 1)
    raise OpError(f'cannot convert {type(obj)} to term and coeff')