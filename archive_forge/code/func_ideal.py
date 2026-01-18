from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible
from sympy.utilities import public
def ideal(self, *gens):
    """
        Generate an ideal of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2)
        <x**2>
        """
    from sympy.polys.agca.ideals import ModuleImplementedIdeal
    return ModuleImplementedIdeal(self, self.free_module(1).submodule(*[[x] for x in gens]))