from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def collected_word(self, word):
    """
        Return the collected form of a word.

        Explanation
        ===========

        A word ``w`` is called collected, if `w = {x_{i_1}}^{a_1} * \\ldots *
        {x_{i_r}}^{a_r}` with `i_1 < i_2< \\ldots < i_r` and `a_j` is in
        `\\{1, \\ldots, {s_j}-1\\}`.

        Otherwise w is uncollected.

        Parameters
        ==========

        word : FreeGroupElement
            An uncollected word.

        Returns
        =======

        word
            A collected word of form `w = {x_{i_1}}^{a_1}, \\ldots,
            {x_{i_r}}^{a_r}` with `i_1, i_2, \\ldots, i_r` and `a_j \\in
            \\{1, \\ldots, {s_j}-1\\}`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1, x2, x3 = free_group("x0, x1, x2, x3")
        >>> word = x3*x2*x1*x0
        >>> collected_word = collector.collected_word(word)
        >>> free_to_perm = {}
        >>> free_group = collector.free_group
        >>> for sym, gen in zip(free_group.symbols, collector.pcgs):
        ...     free_to_perm[sym] = gen
        >>> G1 = PermutationGroup()
        >>> for w in word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G1 = PermutationGroup([perm] + G1.generators)
        >>> G2 = PermutationGroup()
        >>> for w in collected_word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G2 = PermutationGroup([perm] + G2.generators)

        The two are not identical, but they are equivalent:

        >>> G1.equals(G2), G1 == G2
        (True, False)

        See Also
        ========

        minimal_uncollected_subword

        """
    free_group = self.free_group
    while True:
        w = self.minimal_uncollected_subword(word)
        if not w:
            break
        low, high = self.subword_index(word, free_group.dtype(w))
        if low == -1:
            continue
        s1, e1 = w[0]
        if len(w) == 1:
            re = self.relative_order[self.index[s1]]
            q = e1 // re
            r = e1 - q * re
            key = ((w[0][0], re),)
            key = free_group.dtype(key)
            if self.pc_presentation[key]:
                presentation = self.pc_presentation[key].array_form
                sym, exp = presentation[0]
                word_ = ((w[0][0], r), (sym, q * exp))
                word_ = free_group.dtype(word_)
            elif r != 0:
                word_ = ((w[0][0], r),)
                word_ = free_group.dtype(word_)
            else:
                word_ = None
            word = word.eliminate_word(free_group.dtype(w), word_)
        if len(w) == 2 and w[1][1] > 0:
            s2, e2 = w[1]
            s2 = ((s2, 1),)
            s2 = free_group.dtype(s2)
            word_ = self.map_relation(free_group.dtype(w))
            word_ = s2 * word_ ** e1
            word_ = free_group.dtype(word_)
            word = word.substituted_word(low, high, word_)
        elif len(w) == 2 and w[1][1] < 0:
            s2, e2 = w[1]
            s2 = ((s2, 1),)
            s2 = free_group.dtype(s2)
            word_ = self.map_relation(free_group.dtype(w))
            word_ = s2 ** (-1) * word_ ** e1
            word_ = free_group.dtype(word_)
            word = word.substituted_word(low, high, word_)
    return word