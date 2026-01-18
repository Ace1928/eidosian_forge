from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def canonical_free(base, gens, g, num_free):
    """
    Canonicalization of a tensor with respect to free indices
    choosing the minimum with respect to lexicographical ordering
    in the free indices.

    Explanation
    ===========

    ``base``, ``gens``  BSGS for slot permutation group
    ``g``               permutation representing the tensor
    ``num_free``        number of free indices
    The indices must be ordered with first the free indices

    See explanation in double_coset_can_rep
    The algorithm is a variation of the one given in [2].

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import canonical_free
    >>> gens = [[1, 0, 2, 3, 5, 4], [2, 3, 0, 1, 4, 5],[0, 1, 3, 2, 5, 4]]
    >>> gens = [Permutation(h) for h in gens]
    >>> base = [0, 2]
    >>> g = Permutation([2, 1, 0, 3, 4, 5])
    >>> canonical_free(base, gens, g, 4)
    [0, 3, 1, 2, 5, 4]

    Consider the product of Riemann tensors
    ``T = R^{a}_{d0}^{d1,d2}*R_{d2,d1}^{d0,b}``
    The order of the indices is ``[a, b, d0, -d0, d1, -d1, d2, -d2]``
    The permutation corresponding to the tensor is
    ``g = [0, 3, 4, 6, 7, 5, 2, 1, 8, 9]``

    In particular ``a`` is position ``0``, ``b`` is in position ``9``.
    Use the slot symmetries to get `T` is a form which is the minimal
    in lexicographic order in the free indices ``a`` and ``b``, e.g.
    ``-R^{a}_{d0}^{d1,d2}*R^{b,d0}_{d2,d1}`` corresponding to
    ``[0, 3, 4, 6, 1, 2, 7, 5, 9, 8]``

    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, tensor_gens
    >>> base, gens = riemann_bsgs
    >>> size, sbase, sgens = tensor_gens(base, gens, [[], []], 0)
    >>> g = Permutation([0, 3, 4, 6, 7, 5, 2, 1, 8, 9])
    >>> canonical_free(sbase, [Permutation(h) for h in sgens], g, 2)
    [0, 3, 4, 6, 1, 2, 7, 5, 9, 8]
    """
    g = g.array_form
    size = len(g)
    if not base:
        return g[:]
    transversals = get_transversals(base, gens)
    for x in sorted(g[:-2]):
        if x not in base:
            base.append(x)
    h = g
    for i, transv in enumerate(transversals):
        h_i = [size] * num_free
        s = None
        for sk in transv.values():
            h1 = _af_rmul(h, sk)
            hi = [h1.index(ix) for ix in range(num_free)]
            if hi < h_i:
                h_i = hi
                s = sk
        if s:
            h = _af_rmul(h, s)
    return h