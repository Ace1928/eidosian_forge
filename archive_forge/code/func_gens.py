from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
@property
def gens(self):
    """
        Return generators for ``self``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)
        [x, y, x**2 + y]
        """
    return (x[0] for x in self._module.gens)