from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def is_whole_ring(self):
    """
        Return True if ``self`` is the whole ring, i.e. one generator is a unit.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ, ilex
        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()
        False
        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()
        True
        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()
        True
        """
    return self._module.is_full_module()