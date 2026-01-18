from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def dihedral(n):
    """
    Generates the dihedral group of order 2n, Dn.

    The result is given as a subgroup of Sn, except for the special cases n=1
    (the group S2) and n=2 (the Klein 4-group) where that's not possible
    and embeddings in S2 and S4 respectively are given.

    Examples
    ========

    >>> from sympy.combinatorics.generators import dihedral
    >>> list(dihedral(3))
    [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]

    See Also
    ========

    cyclic
    """
    if n == 1:
        yield Permutation([0, 1])
        yield Permutation([1, 0])
    elif n == 2:
        yield Permutation([0, 1, 2, 3])
        yield Permutation([1, 0, 3, 2])
        yield Permutation([2, 3, 0, 1])
        yield Permutation([3, 2, 1, 0])
    else:
        gen = list(range(n))
        for i in range(n):
            yield Permutation(gen)
            yield Permutation(gen[::-1])
            gen = rotate_left(gen, 1)