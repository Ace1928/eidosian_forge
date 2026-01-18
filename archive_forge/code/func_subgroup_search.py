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
def subgroup_search(self, prop, base=None, strong_gens=None, tests=None, init_subgroup=None):
    """Find the subgroup of all elements satisfying the property ``prop``.

        Explanation
        ===========

        This is done by a depth-first search with respect to base images that
        uses several tests to prune the search tree.

        Parameters
        ==========

        prop
            The property to be used. Has to be callable on group elements
            and always return ``True`` or ``False``. It is assumed that
            all group elements satisfying ``prop`` indeed form a subgroup.
        base
            A base for the supergroup.
        strong_gens
            A strong generating set for the supergroup.
        tests
            A list of callables of length equal to the length of ``base``.
            These are used to rule out group elements by partial base images,
            so that ``tests[l](g)`` returns False if the element ``g`` is known
            not to satisfy prop base on where g sends the first ``l + 1`` base
            points.
        init_subgroup
            if a subgroup of the sought group is
            known in advance, it can be passed to the function as this
            parameter.

        Returns
        =======

        res
            The subgroup of all elements satisfying ``prop``. The generating
            set for this group is guaranteed to be a strong generating set
            relative to the base ``base``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> S = SymmetricGroup(7)
        >>> prop_even = lambda x: x.is_even
        >>> base, strong_gens = S.schreier_sims_incremental()
        >>> G = S.subgroup_search(prop_even, base=base, strong_gens=strong_gens)
        >>> G.is_subgroup(AlternatingGroup(7))
        True
        >>> _verify_bsgs(G, base, G.generators)
        True

        Notes
        =====

        This function is extremely lengthy and complicated and will require
        some careful attention. The implementation is described in
        [1], pp. 114-117, and the comments for the code here follow the lines
        of the pseudocode in the book for clarity.

        The complexity is exponential in general, since the search process by
        itself visits all members of the supergroup. However, there are a lot
        of tests which are used to prune the search tree, and users can define
        their own tests via the ``tests`` parameter, so in practice, and for
        some computations, it's not terrible.

        A crucial part in the procedure is the frequent base change performed
        (this is line 11 in the pseudocode) in order to obtain a new basic
        stabilizer. The book mentiones that this can be done by using
        ``.baseswap(...)``, however the current implementation uses a more
        straightforward way to find the next basic stabilizer - calling the
        function ``.stabilizer(...)`` on the previous basic stabilizer.

        """

    def get_reps(orbits):
        return [min(orbit, key=lambda x: base_ordering[x]) for orbit in orbits]

    def update_nu(l):
        temp_index = len(basic_orbits[l]) + 1 - len(res_basic_orbits_init_base[l])
        if temp_index >= len(sorted_orbits[l]):
            nu[l] = base_ordering[degree]
        else:
            nu[l] = sorted_orbits[l][temp_index]
    if base is None:
        base, strong_gens = self.schreier_sims_incremental()
    base_len = len(base)
    degree = self.degree
    identity = _af_new(list(range(degree)))
    base_ordering = _base_ordering(base, degree)
    base_ordering.append(degree)
    base_ordering.append(-1)
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    basic_orbits, transversals = _orbits_transversals_from_bsgs(base, strong_gens_distr)
    if init_subgroup is None:
        init_subgroup = PermutationGroup([identity])
    if tests is None:
        trivial_test = lambda x: True
        tests = []
        for i in range(base_len):
            tests.append(trivial_test)
    res = init_subgroup
    f = base_len - 1
    l = base_len - 1
    res_base = base[:]
    res_base, res_strong_gens = res.schreier_sims_incremental(base=res_base)
    res_strong_gens_distr = _distribute_gens_by_base(res_base, res_strong_gens)
    res_generators = res.generators
    res_basic_orbits_init_base = [_orbit(degree, res_strong_gens_distr[i], res_base[i]) for i in range(base_len)]
    orbit_reps = [None] * base_len
    orbits = _orbits(degree, res_strong_gens_distr[f])
    orbit_reps[f] = get_reps(orbits)
    orbit_reps[f].remove(base[f])
    c = [0] * base_len
    u = [identity] * base_len
    sorted_orbits = [None] * base_len
    for i in range(base_len):
        sorted_orbits[i] = basic_orbits[i][:]
        sorted_orbits[i].sort(key=lambda point: base_ordering[point])
    mu = [None] * base_len
    nu = [None] * base_len
    mu[l] = degree + 1
    update_nu(l)
    computed_words = [identity] * base_len
    while True:
        while l < base_len - 1 and computed_words[l](base[l]) in orbit_reps[l] and (base_ordering[mu[l]] < base_ordering[computed_words[l](base[l])] < base_ordering[nu[l]]) and tests[l](computed_words):
            new_point = computed_words[l](base[l])
            res_base[l] = new_point
            new_stab_gens = _stabilizer(degree, res_strong_gens_distr[l], new_point)
            res_strong_gens_distr[l + 1] = new_stab_gens
            orbits = _orbits(degree, new_stab_gens)
            orbit_reps[l + 1] = get_reps(orbits)
            l += 1
            temp_orbit = [computed_words[l - 1](point) for point in basic_orbits[l]]
            temp_orbit.sort(key=lambda point: base_ordering[point])
            sorted_orbits[l] = temp_orbit
            new_mu = degree + 1
            for i in range(l):
                if base[l] in res_basic_orbits_init_base[i]:
                    candidate = computed_words[i](base[i])
                    if base_ordering[candidate] > base_ordering[new_mu]:
                        new_mu = candidate
            mu[l] = new_mu
            update_nu(l)
            c[l] = 0
            temp_point = sorted_orbits[l][c[l]]
            gamma = computed_words[l - 1]._array_form.index(temp_point)
            u[l] = transversals[l][gamma]
            computed_words[l] = rmul(computed_words[l - 1], u[l])
        g = computed_words[l]
        temp_point = g(base[l])
        if l == base_len - 1 and base_ordering[mu[l]] < base_ordering[temp_point] < base_ordering[nu[l]] and (temp_point in orbit_reps[l]) and tests[l](computed_words) and prop(g):
            res_generators.append(g)
            res_base = base[:]
            res_strong_gens.append(g)
            res_strong_gens_distr = _distribute_gens_by_base(res_base, res_strong_gens)
            res_basic_orbits_init_base = [_orbit(degree, res_strong_gens_distr[i], res_base[i]) for i in range(base_len)]
            orbit_reps[f] = get_reps(orbits)
            l = f
        while l >= 0 and c[l] == len(basic_orbits[l]) - 1:
            l = l - 1
        if l == -1:
            return PermutationGroup(res_generators)
        if l < f:
            f = l
            c[l] = 0
            temp_orbits = _orbits(degree, res_strong_gens_distr[f])
            orbit_reps[f] = get_reps(temp_orbits)
            mu[l] = degree + 1
            temp_index = len(basic_orbits[l]) + 1 - len(res_basic_orbits_init_base[l])
            if temp_index >= len(sorted_orbits[l]):
                nu[l] = base_ordering[degree]
            else:
                nu[l] = sorted_orbits[l][temp_index]
        c[l] += 1
        if l == 0:
            gamma = sorted_orbits[l][c[l]]
        else:
            gamma = computed_words[l - 1]._array_form.index(sorted_orbits[l][c[l]])
        u[l] = transversals[l][gamma]
        if l == 0:
            computed_words[l] = u[l]
        else:
            computed_words[l] = rmul(computed_words[l - 1], u[l])