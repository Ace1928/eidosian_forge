import re
import warnings
from enum import Enum
from math import gcd
def as_apply(func, *args, **kwargs):
    """Return object as APPLY expression (function call, constructor, etc.)
    """
    return Expr(Op.APPLY, (func, tuple(map(as_expr, args)), dict(((k, as_expr(v)) for k, v in kwargs.items()))))