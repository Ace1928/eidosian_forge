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
def centralizer(self, other):
    """
        Return the centralizer of a group/set/element.

        Explanation
        ===========

        The centralizer of a set of permutations ``S`` inside
        a group ``G`` is the set of elements of ``G`` that commute with all
        elements of ``S``::

            `C_G(S) = \\{ g \\in G | gs = sg \\forall s \\in S\\}` ([10])

        Usually, ``S`` is a subset of ``G``, but if ``G`` is a proper subgroup of
        the full symmetric group, we allow for ``S`` to have elements outside
        ``G``.

        It is naturally a subgroup of ``G``; the centralizer of a permutation
        group is equal to the centralizer of any set of generators for that
        group, since any element commuting with the generators commutes with
        any product of the  generators.

        Parameters
        ==========

        other
            a permutation group/list of permutations/single permutation

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> S = SymmetricGroup(6)
        >>> C = CyclicGroup(6)
        >>> H = S.centralizer(C)
        >>> H.is_subgroup(C)
        True

        See Also
        ========

        subgroup_search

        Notes
        =====

        The implementation is an application of ``.subgroup_search()`` with
        tests using a specific base for the group ``G``.

        """
    if hasattr(other, 'generators'):
        if other.is_trivial or self.is_trivial:
            return self
        degree = self.degree
        identity = _af_new(list(range(degree)))
        orbits = other.orbits()
        num_orbits = len(orbits)
        orbits.sort(key=lambda x: -len(x))
        long_base = []
        orbit_reps = [None] * num_orbits
        orbit_reps_indices = [None] * num_orbits
        orbit_descr = [None] * degree
        for i in range(num_orbits):
            orbit = list(orbits[i])
            orbit_reps[i] = orbit[0]
            orbit_reps_indices[i] = len(long_base)
            for point in orbit:
                orbit_descr[point] = i
            long_base = long_base + orbit
        base, strong_gens = self.schreier_sims_incremental(base=long_base)
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        i = 0
        for i in range(len(base)):
            if strong_gens_distr[i] == [identity]:
                break
        base = base[:i]
        base_len = i
        for j in range(num_orbits):
            if base[base_len - 1] in orbits[j]:
                break
        rel_orbits = orbits[:j + 1]
        num_rel_orbits = len(rel_orbits)
        transversals = [None] * num_rel_orbits
        for j in range(num_rel_orbits):
            rep = orbit_reps[j]
            transversals[j] = dict(other.orbit_transversal(rep, pairs=True))
        trivial_test = lambda x: True
        tests = [None] * base_len
        for l in range(base_len):
            if base[l] in orbit_reps:
                tests[l] = trivial_test
            else:

                def test(computed_words, l=l):
                    g = computed_words[l]
                    rep_orb_index = orbit_descr[base[l]]
                    rep = orbit_reps[rep_orb_index]
                    im = g._array_form[base[l]]
                    im_rep = g._array_form[rep]
                    tr_el = transversals[rep_orb_index][base[l]]
                    return im == tr_el._array_form[im_rep]
                tests[l] = test

        def prop(g):
            return [rmul(g, gen) for gen in other.generators] == [rmul(gen, g) for gen in other.generators]
        return self.subgroup_search(prop, base=base, strong_gens=strong_gens, tests=tests)
    elif hasattr(other, '__getitem__'):
        gens = list(other)
        return self.centralizer(PermutationGroup(gens))
    elif hasattr(other, 'array_form'):
        return self.centralizer(PermutationGroup([other]))