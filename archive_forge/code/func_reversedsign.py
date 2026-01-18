from typing import Optional
from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not
@property
def reversedsign(self):
    """
        Try to return the relationship with signs reversed.
        """
    revfunc = self.function.reversed
    if revfunc is None:
        return self
    if not any((side.kind is BooleanKind for side in self.arguments)):
        return revfunc(-self.lhs, -self.rhs)
    return self