from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
@public
def prime_decomp(p, T=None, ZK=None, dK=None, radical=None):
    """
    Compute the decomposition of rational prime *p* in a number field.

    Explanation
    ===========

    Ordinarily this should be accessed through the
    :py:meth:`~.AlgebraicField.primes_above` method of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x, theta
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.algebraic_field((T, theta))
    >>> print(K.primes_above(2))
    [[ (2, x**2 + 1) e=1, f=1 ], [ (2, (x**2 + 3*x + 2)/2) e=1, f=1 ],
     [ (2, (3*x**2 + 3*x)/2) e=1, f=1 ]]

    Parameters
    ==========

    p : int
        The rational prime whose decomposition is desired.

    T : :py:class:`~.Poly`, optional
        Monic irreducible polynomial defining the number field $K$ in which to
        factor. NOTE: at least one of *T* or *ZK* must be provided.

    ZK : :py:class:`~.Submodule`, optional
        The maximal order for $K$, if already known.
        NOTE: at least one of *T* or *ZK* must be provided.

    dK : int, optional
        The discriminant of the field $K$, if already known.

    radical : :py:class:`~.Submodule`, optional
        The nilradical mod *p* in the integers of $K$, if already known.

    Returns
    =======

    List of :py:class:`~.PrimeIdeal` instances.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 6.2.9.)

    """
    if T is None and ZK is None:
        raise ValueError('At least one of T or ZK must be provided.')
    if ZK is not None:
        _check_formal_conditions_for_maximal_order(ZK)
    if T is None:
        T = ZK.parent.T
    radicals = {}
    if dK is None or ZK is None:
        ZK, dK = round_two(T, radicals=radicals)
    dT = T.discriminant()
    f_squared = dT // dK
    if f_squared % p != 0:
        return _prime_decomp_easy_case(p, ZK)
    radical = radical or radicals.get(p) or nilradical_mod_p(ZK, p)
    stack = [radical]
    primes = []
    while stack:
        I = stack.pop()
        N, G = _prime_decomp_compute_kernel(I, p, ZK)
        if N.n == 1:
            P = _prime_decomp_maximal_ideal(I, p, ZK)
            primes.append(P)
        else:
            I1, I2 = _prime_decomp_split_ideal(I, p, N, G, ZK)
            stack.extend([I1, I2])
    return primes