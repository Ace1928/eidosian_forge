from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence
def minimal(self):
    """
        Return minimal Weierstrass equation.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve

        >>> e1 = EllipticCurve(-10, -20, 0, -1, 1)
        >>> e1.minimal()
        E(QQ): Eq(y**2*z, x**3 - 13392*x*z**2 - 1080432*z**3)

        """
    char = self.characteristic
    if char == 2:
        return self
    if char == 3:
        return EllipticCurve(self._b4 / 2, self._b6 / 4, a2=self._b2 / 4, modulus=self.modulus)
    c4 = self._b2 ** 2 - 24 * self._b4
    c6 = -self._b2 ** 3 + 36 * self._b2 * self._b4 - 216 * self._b6
    return EllipticCurve(-27 * c4, -54 * c6, modulus=self.modulus)