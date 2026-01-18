from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def restrict_codomain(self, sm):
    """
        Return ``self``, with codomain restricted to to ``sm``.

        Here ``sm`` has to be a submodule of ``self.codomain`` containing the
        image.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_codomain(F.submodule([1, 0]))
        Matrix([
        [1, x], : QQ[x]**2 -> <[1, 0]>
        [0, 0]])
        """
    if not sm.is_submodule(self.image()):
        raise ValueError('the image %s must contain sm, got %s' % (self.image(), sm))
    if sm == self.codomain:
        return self
    return self._restrict_codomain(sm)