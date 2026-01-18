from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
def kahane_simplify(expression):
    """
    This function cancels contracted elements in a product of four
    dimensional gamma matrices, resulting in an expression equal to the given
    one, without the contracted gamma matrices.

    Parameters
    ==========

    `expression`    the tensor expression containing the gamma matrices to simplify.

    Notes
    =====

    If spinor indices are given, the matrices must be given in
    the order given in the product.

    Algorithm
    =========

    The idea behind the algorithm is to use some well-known identities,
    i.e., for contractions enclosing an even number of `\\gamma` matrices

    `\\gamma^\\mu \\gamma_{a_1} \\cdots \\gamma_{a_{2N}} \\gamma_\\mu = 2 (\\gamma_{a_{2N}} \\gamma_{a_1} \\cdots \\gamma_{a_{2N-1}} + \\gamma_{a_{2N-1}} \\cdots \\gamma_{a_1} \\gamma_{a_{2N}} )`

    for an odd number of `\\gamma` matrices

    `\\gamma^\\mu \\gamma_{a_1} \\cdots \\gamma_{a_{2N+1}} \\gamma_\\mu = -2 \\gamma_{a_{2N+1}} \\gamma_{a_{2N}} \\cdots \\gamma_{a_{1}}`

    Instead of repeatedly applying these identities to cancel out all contracted indices,
    it is possible to recognize the links that would result from such an operation,
    the problem is thus reduced to a simple rearrangement of free gamma matrices.

    Examples
    ========

    When using, always remember that the original expression coefficient
    has to be handled separately

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.physics.hep.gamma_matrices import kahane_simplify
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)
    >>> ta = G(i0)*G(-i0)
    >>> kahane_simplify(ta)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> tb = G(i0)*G(i1)*G(-i0)
    >>> kahane_simplify(tb)
    -2*GammaMatrix(i1)
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])

    If there are no contractions, the same expression is returned

    >>> tc = G(i0)*G(i1)
    >>> kahane_simplify(tc)
    GammaMatrix(i0)*GammaMatrix(i1)

    References
    ==========

    [1] Algorithm for Reducing Contracted Products of gamma Matrices,
    Joseph Kahane, Journal of Mathematical Physics, Vol. 9, No. 10, October 1968.
    """
    if isinstance(expression, Mul):
        return expression
    if isinstance(expression, TensAdd):
        return TensAdd(*[kahane_simplify(arg) for arg in expression.args])
    if isinstance(expression, Tensor):
        return expression
    assert isinstance(expression, TensMul)
    gammas = expression.args
    for gamma in gammas:
        assert gamma.component == GammaMatrix
    free = expression.free
    dum = []
    for dum_pair in expression.dum:
        if expression.index_types[dum_pair[0]] == LorentzIndex:
            dum.append((dum_pair[0], dum_pair[1]))
    dum = sorted(dum)
    if len(dum) == 0:
        return expression
    first_dum_pos = min(map(min, dum))
    total_number = len(free) + len(dum) * 2
    number_of_contractions = len(dum)
    free_pos = [None] * total_number
    for i in free:
        free_pos[i[1]] = i[0]
    index_is_free = [False] * total_number
    for i, indx in enumerate(free):
        index_is_free[indx[1]] = True
    links = {i: [] for i in range(first_dum_pos, total_number)}
    cum_sign = -1
    cum_sign_list = [None] * total_number
    block_free_count = 0
    resulting_coeff = S.One
    resulting_indices = [[]]
    connected_components = 1
    for i, is_free in enumerate(index_is_free):
        if i < first_dum_pos:
            continue
        if is_free:
            block_free_count += 1
            if block_free_count > 1:
                links[i - 1].append(i)
                links[i].append(i - 1)
        else:
            cum_sign *= 1 if block_free_count % 2 else -1
            if block_free_count == 0 and i != first_dum_pos:
                if cum_sign == -1:
                    links[-1 - i] = [-1 - i + 1]
                    links[-1 - i + 1] = [-1 - i]
            if i - cum_sign in links:
                if i != first_dum_pos:
                    links[i].append(i - cum_sign)
                if block_free_count != 0:
                    if i - cum_sign < len(index_is_free):
                        if index_is_free[i - cum_sign]:
                            links[i - cum_sign].append(i)
            block_free_count = 0
        cum_sign_list[i] = cum_sign
    for i in dum:
        pos1 = i[0]
        pos2 = i[1]
        links[pos1].append(pos2)
        links[pos2].append(pos1)
        linkpos1 = pos1 + cum_sign_list[pos1]
        linkpos2 = pos2 + cum_sign_list[pos2]
        if linkpos1 >= total_number:
            continue
        if linkpos2 >= total_number:
            continue
        if linkpos1 < first_dum_pos:
            continue
        if linkpos2 < first_dum_pos:
            continue
        if -1 - linkpos1 in links:
            linkpos1 = -1 - linkpos1
        if -1 - linkpos2 in links:
            linkpos2 = -1 - linkpos2
        if linkpos1 >= 0 and (not index_is_free[linkpos1]):
            linkpos1 = pos1
        if linkpos2 >= 0 and (not index_is_free[linkpos2]):
            linkpos2 = pos2
        if linkpos2 not in links[linkpos1]:
            links[linkpos1].append(linkpos2)
        if linkpos1 not in links[linkpos2]:
            links[linkpos2].append(linkpos1)
    pointer = first_dum_pos
    previous_pointer = 0
    while True:
        if pointer in links:
            next_ones = links.pop(pointer)
        else:
            break
        if previous_pointer in next_ones:
            next_ones.remove(previous_pointer)
        previous_pointer = pointer
        if next_ones:
            pointer = next_ones[0]
        else:
            break
        if pointer == previous_pointer:
            break
        if pointer >= 0 and free_pos[pointer] is not None:
            for ri in resulting_indices:
                ri.append(free_pos[pointer])
    while links:
        connected_components += 1
        pointer = min(links.keys())
        previous_pointer = pointer
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break
            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)
            previous_pointer = pointer
            if next_ones:
                pointer = next_ones[0]
            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    resulting_coeff *= 2 ** number_of_contractions
    resulting_indices = [free_pos[0:first_dum_pos] + ri for ri in resulting_indices]
    resulting_expr = S.Zero
    for i in resulting_indices:
        temp_expr = S.One
        for j in i:
            temp_expr *= GammaMatrix(j)
        resulting_expr += temp_expr
    t = resulting_coeff * resulting_expr
    t1 = None
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    elif isinstance(t, TensMul):
        t1 = t
    if t1:
        pass
    else:
        t = eye(4) * t
    return t