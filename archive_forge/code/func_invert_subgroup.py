import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def invert_subgroup(self, H):
    """
        Return the subgroup of the domain that is the inverse image
        of the subgroup ``H`` of the homomorphism image

        """
    if not H.is_subgroup(self.image()):
        raise ValueError('Given H is not a subgroup of the image')
    gens = []
    P = PermutationGroup(self.image().identity)
    for h in H.generators:
        h_i = self.invert(h)
        if h_i not in P:
            gens.append(h_i)
            P = PermutationGroup(gens)
        for k in self.kernel().generators:
            if k * h_i not in P:
                gens.append(k * h_i)
                P = PermutationGroup(gens)
    return P