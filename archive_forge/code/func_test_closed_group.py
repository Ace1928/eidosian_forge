from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
from sympy.utilities.exceptions import sympy_deprecation_warning
from ..predicates.common import CommutativePredicate, IsTruePredicate
def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
    return _fuzzy_group((ask(key(a), assumptions) for a in expr.args), quick_exit=True)