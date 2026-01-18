from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
def points_x(self, x):
    """Returns points on with curve where xcoordinate = x"""
    pt = []
    if self._domain == QQ:
        for y in solve(self._eq.subs(self.x, x)):
            pt.append((x, y))
    congruence_eq = (self._eq.lhs - self._eq.rhs).subs({self.x: x, self.z: 1})
    for y in polynomial_congruence(congruence_eq, self.characteristic):
        pt.append((x, y))
    return pt