from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def dummy_sgs(dummies, sym, n):
    """
    Return the strong generators for dummy indices.

    Parameters
    ==========

    dummies : List of dummy indices.
        `dummies[2k], dummies[2k+1]` are paired indices.
        In base form, the dummy indices are always in
        consecutive positions.
    sym : symmetry under interchange of contracted dummies::
        * None  no symmetry
        * 0     commuting
        * 1     anticommuting

    n : number of indices

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import dummy_sgs
    >>> dummy_sgs(list(range(2, 8)), 0, 8)
    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],
     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]
    """
    if len(dummies) > n:
        raise ValueError('List too large')
    res = []
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            a[j], a[j + 1] = (a[j + 1], a[j])
            res.append(a)
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = (a[j + 2], a[j + 3], a[j], a[j + 1])
        res.append(a)
    return res