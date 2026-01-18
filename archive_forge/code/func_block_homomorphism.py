import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def block_homomorphism(group, blocks):
    """
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the block system ``blocks``. The latter should be
    of the same form as returned by the ``minimal_block`` method for
    permutation groups, namely a list of length ``group.degree`` where
    the i-th entry is a representative of the block i belongs to.

    """
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup
    n = len(blocks)
    m = 0
    p = []
    b = [None] * n
    for i in range(n):
        if blocks[i] == i:
            p.append(i)
            b[i] = m
            m += 1
    for i in range(n):
        b[i] = b[blocks[i]]
    codomain = SymmetricGroup(m)
    identity = range(m)
    images = {g: Permutation([b[p[i] ^ g] for i in identity]) for g in group.generators}
    H = GroupHomomorphism(group, codomain, images)
    return H