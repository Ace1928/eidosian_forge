from __future__ import annotations
from .basic import Atom, Basic
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol
def is_lt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly less than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_ge(lhs, rhs, assumptions))