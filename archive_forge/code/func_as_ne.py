import re
import warnings
from enum import Enum
from math import gcd
def as_ne(left, right):
    return Expr(Op.RELATIONAL, (RelOp.NE, left, right))