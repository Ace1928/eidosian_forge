from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def is_isomorphism(self):
    """
        Return True if ``self`` is an isomorphism.

        That is, check if every element of the codomain has precisely one
        preimage. Equivalently, ``self`` is both injective and surjective.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h = h.restrict_codomain(h.image())
        >>> h.is_isomorphism()
        False
        >>> h.quotient_domain(h.kernel()).is_isomorphism()
        True
        """
    return self.is_injective() and self.is_surjective()