from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def rsa_public_key(*args, **kwargs):
    """Return the RSA *public key* pair, `(n, e)`

    Parameters
    ==========

    args : naturals
        If specified as `p, q, e` where `p` and `q` are distinct primes
        and `e` is a desired public exponent of the RSA, `n = p q` and
        `e` will be verified against the totient
        `\\phi(n)` (Euler totient) or `\\lambda(n)` (Carmichael totient)
        to be `\\gcd(e, \\phi(n)) = 1` or `\\gcd(e, \\lambda(n)) = 1`.

        If specified as `p_1, p_2, \\dots, p_n, e` where
        `p_1, p_2, \\dots, p_n` are specified as primes,
        and `e` is specified as a desired public exponent of the RSA,
        it will be able to form a multi-prime RSA, which is a more
        generalized form of the popular 2-prime RSA.

        It can also be possible to form a single-prime RSA by specifying
        the argument as `p, e`, which can be considered a trivial case
        of a multiprime RSA.

        Furthermore, it can be possible to form a multi-power RSA by
        specifying two or more pairs of the primes to be same.
        However, unlike the two-distinct prime RSA or multi-prime
        RSA, not every numbers in the complete residue system
        (`\\mathbb{Z}_n`) will be decryptable since the mapping
        `\\mathbb{Z}_{n} \\rightarrow \\mathbb{Z}_{n}`
        will not be bijective.
        (Only except for the trivial case when
        `e = 1`
        or more generally,

        .. math::
            e \\in \\left \\{ 1 + k \\lambda(n)
            \\mid k \\in \\mathbb{Z} \\land k \\geq 0 \\right \\}

        when RSA reduces to the identity.)
        However, the RSA can still be decryptable for the numbers in the
        reduced residue system (`\\mathbb{Z}_n^{\\times}`), since the
        mapping
        `\\mathbb{Z}_{n}^{\\times} \\rightarrow \\mathbb{Z}_{n}^{\\times}`
        can still be bijective.

        If you pass a non-prime integer to the arguments
        `p_1, p_2, \\dots, p_n`, the particular number will be
        prime-factored and it will become either a multi-prime RSA or a
        multi-power RSA in its canonical form, depending on whether the
        product equals its radical or not.
        `p_1 p_2 \\dots p_n = \\text{rad}(p_1 p_2 \\dots p_n)`

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient `\\phi(n)` which is
        :meth:`sympy.ntheory.factor_.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient `\\lambda(n)`
        which is :meth:`sympy.ntheory.factor_.reduced_totient` in SymPy.

        Unlike private key generation, this is a trivial keyword for
        public key generation because
        `\\gcd(e, \\phi(n)) = 1 \\iff \\gcd(e, \\lambda(n)) = 1`.

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA public key at the index
        specified at `0, 1, 2, \\dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        Similarly to the non-uniquenss of a RSA private key as described
        in the ``index`` parameter documentation in
        :meth:`rsa_private_key`, RSA public key is also not unique and
        there is an infinite number of RSA public exponents which
        can behave in the same manner.

        From any given RSA public exponent `e`, there are can be an
        another RSA public exponent `e + k \\lambda(n)` where `k` is an
        integer, `\\lambda` is a Carmichael's totient function.

        However, considering only the positive cases, there can be
        a principal solution of a RSA public exponent `e_0` in
        `0 < e_0 < \\lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `e_0 + k \\lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA public key can have.

        An example of computing any arbitrary RSA public key:

        >>> from sympy.crypto.crypto import rsa_public_key
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 17)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 797)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1577)

    multipower : bool, optional
        Any pair of non-distinct primes found in the RSA specification
        will restrict the domain of the cryptosystem, as noted in the
        explanation of the parameter ``args``.

        SymPy RSA key generator may give a warning before dispatching it
        as a multi-power RSA, however, you can disable the warning if
        you pass ``True`` to this keyword.

    Returns
    =======

    (n, e) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `e` is relatively prime (coprime) to the Euler totient
        `\\phi(n)`.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_public_key

    A public key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_public_key(p, q, e)
    (15, 7)
    >>> rsa_public_key(p, q, 30)
    False

    A public key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_public_key(*args)
    (30030, 7)

    Notes
    =====

    Although the RSA can be generalized over any modulus `n`, using
    two large primes had became the most popular specification because a
    product of two large primes is usually the hardest to factor
    relatively to the digits of `n` can have.

    However, it may need further understanding of the time complexities
    of each prime-factoring algorithms to verify the claim.

    See Also
    ========

    rsa_private_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=True, private=False, **kwargs)