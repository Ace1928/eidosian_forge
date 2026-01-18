from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
def quotient_module(self, other, **opts):
    """
        Return a quotient module.

        This is the same as taking a submodule of a quotient of the containing
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> S1 = F.submodule([x, 1])
        >>> S2 = F.submodule([x**2, x])
        >>> S1.quotient_module(S2)
        <[x, 1] + <[x**2, x]>>

        Or more coincisely, using the overloaded division operator:

        >>> F.submodule([x, 1]) / [(x**2, x)]
        <[x, 1] + <[x**2, x]>>
        """
    if not self.is_submodule(other):
        raise ValueError('%s not a submodule of %s' % (other, self))
    return SubQuotientModule(self.gens, self.container.quotient_module(other), **opts)