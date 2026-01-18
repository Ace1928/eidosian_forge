from __future__ import annotations
from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent
import string
import re as _re
import random
from itertools import product
from typing import Any
def numbered_string_incr(s, start=0):
    if not s:
        return str(start)
    i = len(s) - 1
    while i != -1:
        if not s[i].isdigit():
            break
        i -= 1
    n = str(int(s[i + 1:] or start - 1) + 1)
    return s[:i + 1] + n