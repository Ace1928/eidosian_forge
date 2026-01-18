from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def canonicalize_naive(g, dummies, sym, *v):
    """
    Canonicalize tensor formed by tensors of the different types.

    Explanation
    ===========

    sym_i symmetry under exchange of two component tensors of type `i`
          None  no symmetry
          0     commuting
          1     anticommuting

    Parameters
    ==========

    g : Permutation representing the tensor.
    dummies : List of dummy indices.
    msym : Symmetry of the metric.
    v : A list of (base_i, gens_i, n_i, sym_i) for tensors of type `i`.
        base_i, gens_i BSGS for tensors of this type
        n_i  number of tensors of type `i`

    Returns
    =======

    Returns 0 if the tensor is zero, else returns the array form of
    the permutation representing the canonical form of the tensor.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import canonicalize_naive
    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> from sympy.combinatorics import Permutation
    >>> g = Permutation([1, 3, 2, 0, 4, 5])
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> canonicalize_naive(g, [2, 3], 0, (base2, gens2, 2, 0))
    [0, 2, 1, 3, 4, 5]
    """
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.combinatorics.tensor_can import gens_products, dummy_sgs
    from sympy.combinatorics.permutations import _af_rmul
    v1 = []
    for i in range(len(v)):
        base_i, gens_i, n_i, sym_i = v[i]
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
    size, sbase, sgens = gens_products(*v1)
    dgens = dummy_sgs(dummies, sym, size - 2)
    if isinstance(sym, int):
        num_types = 1
        dummies = [dummies]
        sym = [sym]
    else:
        num_types = len(sym)
    dgens = []
    for i in range(num_types):
        dgens.extend(dummy_sgs(dummies[i], sym[i], size - 2))
    S = PermutationGroup(sgens)
    D = PermutationGroup([Permutation(x) for x in dgens])
    dlist = list(D.generate(af=True))
    g = g.array_form
    st = set()
    for s in S.generate(af=True):
        h = _af_rmul(g, s)
        for d in dlist:
            q = tuple(_af_rmul(d, h))
            st.add(q)
    a = list(st)
    a.sort()
    prev = (0,) * size
    for h in a:
        if h[:-2] == prev[:-2]:
            if h[-1] != prev[-1]:
                return 0
        prev = h
    return list(a[0])