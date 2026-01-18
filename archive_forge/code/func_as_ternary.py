import re
import warnings
from enum import Enum
from math import gcd
def as_ternary(cond, expr1, expr2):
    """Return object as TERNARY expression (cond?expr1:expr2).
    """
    return Expr(Op.TERNARY, (cond, expr1, expr2))