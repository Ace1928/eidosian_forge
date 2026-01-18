import re
import warnings
from enum import Enum
from math import gcd
def as_factors(obj):
    """Return expression as FACTORS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.FACTORS:
            return obj
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                if coeff == 1:
                    return Expr(Op.FACTORS, {term: 1})
                return Expr(Op.FACTORS, {term: 1, Expr.number(coeff): 1})
        if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV and (not obj.data[2]):
            return Expr(Op.FACTORS, {obj.data[1][0]: 1, obj.data[1][1]: -1})
        return Expr(Op.FACTORS, {obj: 1})
    raise OpError(f'cannot convert {type(obj)} to terms Expr')