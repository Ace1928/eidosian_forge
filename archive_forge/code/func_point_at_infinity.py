from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
@staticmethod
def point_at_infinity(curve):
    return EllipticCurvePoint(0, 1, 0, curve)