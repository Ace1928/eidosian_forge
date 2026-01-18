from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def quotient(self, J, **opts):
    """
        Compute the ideal quotient of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J = \\{x \\in R | xJ \\subset I \\}`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x*y).quotient(R.ideal(x))
        <y>
        """
    self._check_ideal(J)
    return self._quotient(J, **opts)