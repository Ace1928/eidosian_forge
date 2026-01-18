from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
@property
def is_singular(self):
    """
        Return True if curve discriminant is equal to zero.
        """
    return self.discriminant == 0