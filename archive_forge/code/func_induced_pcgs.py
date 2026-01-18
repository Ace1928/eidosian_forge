from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def induced_pcgs(self, gens):
    """

        Parameters
        ==========

        gens : list
            A list of generators on which polycyclic subgroup
            is to be defined.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
    z = [1] * len(self.pcgs)
    G = gens
    while G:
        g = G.pop(0)
        h = self._sift(z, g)
        d = self.depth(h)
        if d < len(self.pcgs):
            for gen in z:
                if gen != 1:
                    G.append(h ** (-1) * gen ** (-1) * h * gen)
            z[d - 1] = h
    z = [gen for gen in z if gen != 1]
    return z