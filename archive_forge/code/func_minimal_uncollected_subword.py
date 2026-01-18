from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def minimal_uncollected_subword(self, word):
    """
        Returns the minimal uncollected subwords.

        Explanation
        ===========

        A word ``v`` defined on generators in ``X`` is a minimal
        uncollected subword of the word ``w`` if ``v`` is a subword
        of ``w`` and it has one of the following form

        * `v = {x_{i+1}}^{a_j}x_i`

        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`

        * `v = {x_i}^{a_j}`

        for `a_j` not in `\\{1, \\ldots, s-1\\}`. Where, ``s`` is the power
        exponent of the corresponding generator.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> collector.minimal_uncollected_subword(word)
        ((x2, 2),)

        """
    if not word:
        return None
    array = word.array_form
    re = self.relative_order
    index = self.index
    for i in range(len(array)):
        s1, e1 = array[i]
        if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]] - 1):
            return ((s1, e1),)
    for i in range(len(array) - 1):
        s1, e1 = array[i]
        s2, e2 = array[i + 1]
        if index[s1] > index[s2]:
            e = 1 if e2 > 0 else -1
            return ((s1, e1), (s2, e))
    return None