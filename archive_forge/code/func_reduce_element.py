from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
def reduce_element(self, elt):
    """
        If this submodule $B$ has defining matrix $W$ in square, maximal-rank
        Hermite normal form, then, given an element $x$ of the parent module
        $A$, we produce an element $y \\in A$ such that $x - y \\in B$, and the
        $i$th coordinate of $y$ satisfies $0 \\leq y_i < w_{i,i}$. This
        representative $y$ is unique, in the sense that every element of
        the coset $x + B$ reduces to it under this procedure.

        Explanation
        ===========

        In the special case where $A$ is a power basis for a number field $K$,
        and $B$ is a submodule representing an ideal $I$, this operation
        represents one of a few important ways of reducing an element of $K$
        modulo $I$ to obtain a "small" representative. See [Cohen00]_ Section
        1.4.3.

        Examples
        ========

        >>> from sympy import QQ, Poly, symbols
        >>> t = symbols('t')
        >>> k = QQ.alg_field_from_poly(Poly(t**3 + t**2 - 2*t + 8))
        >>> Zk = k.maximal_order()
        >>> A = Zk.parent
        >>> B = (A(2) - 3*A(0))*Zk
        >>> B.reduce_element(A(2))
        [3, 0, 0]

        Parameters
        ==========

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Returns
        =======

        elt : :py:class:`~.ModuleElement`
            An element of this submodule's parent module.

        Raises
        ======

        NotImplementedError
            If the given :py:class:`~.ModuleElement` does not belong to this
            submodule's parent module.
        StructureError
            If this submodule's defining matrix is not in square, maximal-rank
            Hermite normal form.

        References
        ==========

        .. [Cohen00] Cohen, H. *Advanced Topics in Computational Number
           Theory.*

        """
    if not elt.module == self.parent:
        raise NotImplementedError
    if not self.is_sq_maxrank_HNF():
        msg = 'Reduction not implemented unless matrix square max-rank HNF'
        raise StructureError(msg)
    B = self.basis_element_pullbacks()
    a = elt
    for i in range(self.n - 1, -1, -1):
        b = B[i]
        q = a.coeffs[i] * b.denom // (b.coeffs[i] * a.denom)
        a -= q * b
    return a