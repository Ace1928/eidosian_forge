from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def is_surjective(self):
    """
        Return True if ``self`` is surjective.

        That is, check if every element of the codomain has at least one
        preimage.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_surjective()
        False
        >>> h.restrict_codomain(h.image()).is_surjective()
        True
        """
    return self.image() == self.codomain