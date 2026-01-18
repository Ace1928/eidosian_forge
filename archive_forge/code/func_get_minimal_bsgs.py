from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def get_minimal_bsgs(base, gens):
    """
    Compute a minimal GSGS

    base, gens BSGS

    If base, gens is a minimal BSGS return it; else return a minimal BSGS
    if it fails in finding one, it returns None

    TODO: use baseswap in the case in which if it fails in finding a
    minimal BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> get_minimal_bsgs(*riemann_bsgs1)
    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
    """
    G = PermutationGroup(gens)
    base, gens = G.schreier_sims_incremental()
    if not _is_minimal_bsgs(base, gens):
        return None
    return (base, gens)