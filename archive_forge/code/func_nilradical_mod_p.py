from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant
def nilradical_mod_p(H, p, q=None):
    """
    Compute the nilradical mod *p* for a given order *H*, and prime *p*.

    Explanation
    ===========

    This is the ideal $I$ in $H/pH$ consisting of all elements some positive
    power of which is zero in this quotient ring, i.e. is a multiple of *p*.

    Parameters
    ==========

    H : :py:class:`~.Submodule`
        The given order.
    p : int
        The rational prime.
    q : int, optional
        If known, the smallest power of *p* that is $>=$ the dimension of *H*.
        If not provided, we compute it here.

    Returns
    =======

    :py:class:`~.Module` representing the nilradical mod *p* in *H*.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
    (See Lemma 6.1.6.)

    """
    n = H.n
    if q is None:
        q = p
        while q < n:
            q *= p
    phi = ModuleEndomorphism(H, lambda x: x ** q)
    return phi.kernel(modulus=p)