from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def is_injective(self):
    """
        Return True if ``self`` is injective.

        That is, check if the elements of the domain are mapped to the same
        codomain element.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_injective()
        False
        >>> h.quotient_domain(h.kernel()).is_injective()
        True
        """
    return self.kernel().is_zero()