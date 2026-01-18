from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible
from sympy.utilities import public
def quotient_ring(self, e):
    """
        Form a quotient ring of ``self``.

        Here ``e`` can be an ideal or an iterable.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).quotient_ring(QQ.old_poly_ring(x).ideal(x**2))
        QQ[x]/<x**2>
        >>> QQ.old_poly_ring(x).quotient_ring([x**2])
        QQ[x]/<x**2>

        The division operator has been overloaded for this:

        >>> QQ.old_poly_ring(x)/[x**2]
        QQ[x]/<x**2>
        """
    from sympy.polys.agca.ideals import Ideal
    from sympy.polys.domains.quotientring import QuotientRing
    if not isinstance(e, Ideal):
        e = self.ideal(*e)
    return QuotientRing(self, e)