from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def sylow_subgroup(self, p):
    """
        Return a p-Sylow subgroup of the group.

        The algorithm is described in [1], Chapter 4, Section 7

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import AlternatingGroup

        >>> D = DihedralGroup(6)
        >>> S = D.sylow_subgroup(2)
        >>> S.order()
        4
        >>> G = SymmetricGroup(6)
        >>> S = G.sylow_subgroup(5)
        >>> S.order()
        5

        >>> G1 = AlternatingGroup(3)
        >>> G2 = AlternatingGroup(5)
        >>> G3 = AlternatingGroup(9)

        >>> S1 = G1.sylow_subgroup(3)
        >>> S2 = G2.sylow_subgroup(3)
        >>> S3 = G3.sylow_subgroup(3)

        >>> len1 = len(S1.lower_central_series())
        >>> len2 = len(S2.lower_central_series())
        >>> len3 = len(S3.lower_central_series())

        >>> len1 == len2
        True
        >>> len1 < len3
        True

        """
    from sympy.combinatorics.homomorphisms import orbit_homomorphism, block_homomorphism
    if not isprime(p):
        raise ValueError('p must be a prime')

    def is_p_group(G):
        m = G.order()
        n = 0
        while m % p == 0:
            m = m / p
            n += 1
            if m == 1:
                return (True, n)
        return (False, n)

    def _sylow_reduce(mu, nu):
        Q = mu.image().sylow_subgroup(p)
        Q = mu.invert_subgroup(Q)
        nu = nu.restrict_to(Q)
        R = nu.image().sylow_subgroup(p)
        return nu.invert_subgroup(R)
    order = self.order()
    if order % p != 0:
        return PermutationGroup([self.identity])
    p_group, n = is_p_group(self)
    if p_group:
        return self
    if self.is_alt_sym():
        return PermutationGroup(self._sylow_alt_sym(p))
    orbits = self.orbits()
    non_p_orbits = [o for o in orbits if len(o) % p != 0 and len(o) != 1]
    if non_p_orbits:
        G = self.stabilizer(list(non_p_orbits[0]).pop())
        return G.sylow_subgroup(p)
    if not self.is_transitive():
        orbits = sorted(orbits, key=len)
        omega1 = orbits.pop()
        omega2 = orbits[0].union(*orbits)
        mu = orbit_homomorphism(self, omega1)
        nu = orbit_homomorphism(self, omega2)
        return _sylow_reduce(mu, nu)
    blocks = self.minimal_blocks()
    if len(blocks) > 1:
        mu = block_homomorphism(self, blocks[0])
        nu = block_homomorphism(self, blocks[1])
        return _sylow_reduce(mu, nu)
    elif len(blocks) == 1:
        block = list(blocks)[0]
        if any((e != 0 for e in block)):
            mu = block_homomorphism(self, block)
            if not is_p_group(mu.image())[0]:
                S = mu.image().sylow_subgroup(p)
                return mu.invert_subgroup(S).sylow_subgroup(p)
    g = self.random()
    g_order = g.order()
    while g_order % p != 0 or g_order == 0:
        g = self.random()
        g_order = g.order()
    g = g ** (g_order // p)
    if order % p ** 2 != 0:
        return PermutationGroup(g)
    C = self.centralizer(g)
    while C.order() % p ** n != 0:
        S = C.sylow_subgroup(p)
        s_order = S.order()
        Z = S.center()
        P = Z._p_elements_group(p)
        h = P.random()
        C_h = self.centralizer(h)
        while C_h.order() % p * s_order != 0:
            h = P.random()
            C_h = self.centralizer(h)
        C = C_h
    return C.sylow_subgroup(p)