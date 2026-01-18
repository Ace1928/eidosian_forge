from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def gens_products(*v):
    """
    Returns size, res_base, res_gens BSGS for n tensors of different types.

    Explanation
    ===========

    v is a sequence of (base_i, gens_i, free_i, sym_i)
    where
    base_i, gens_i  BSGS of tensor of type `i`
    free_i          list of the fixed slots for each of the tensors
                    of type `i`; if there are `n_i` tensors of type `i`
                    and none of them have fixed slots, `free = [[]]*n_i`
    sym   0 (1) if the tensors of type `i` (anti)commute among themselves

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
    >>> base, gens = get_symmetric_group_sgs(2)
    >>> gens_products((base, gens, [[], []], 0))
    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
    >>> gens_products((base, gens, [[1], []], 0))
    (6, [2], [(5)(2 3)])
    """
    res_size, res_base, res_gens = tensor_gens(*v[0])
    for i in range(1, len(v)):
        size, base, gens = tensor_gens(*v[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base, gens, 1)
    res_size = res_gens[0].size
    id_af = list(range(res_size))
    res_gens = [h for h in res_gens if h != id_af]
    if not res_gens:
        res_gens = [id_af]
    return (res_size, res_base, res_gens)