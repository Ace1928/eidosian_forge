from sympy.utilities.misc import as_int
def multinomial_coefficients(m, n):
    """Return a dictionary containing pairs ``{(k1,k2,..,km) : C_kn}``
    where ``C_kn`` are multinomial coefficients such that
    ``n=k1+k2+..+km``.

    Examples
    ========

    >>> from sympy.ntheory import multinomial_coefficients
    >>> multinomial_coefficients(2, 5) # indirect doctest
    {(0, 5): 1, (1, 4): 5, (2, 3): 10, (3, 2): 10, (4, 1): 5, (5, 0): 1}

    Notes
    =====

    The algorithm is based on the following result:

    .. math::
        \\binom{n}{k_1, \\ldots, k_m} =
        \\frac{k_1 + 1}{n - k_1} \\sum_{i=2}^m \\binom{n}{k_1 + 1, \\ldots, k_i - 1, \\ldots}

    Code contributed to Sage by Yann Laigle-Chapuy, copied with permission
    of the author.

    See Also
    ========

    binomial_coefficients_list, binomial_coefficients
    """
    m = as_int(m)
    n = as_int(n)
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(multinomial_coefficients_iterator(m, n))
    t = [n] + [0] * (m - 1)
    r = {tuple(t): 1}
    if n:
        j = 0
    else:
        j = m
    while j < m - 1:
        tj = t[j]
        if j:
            t[j] = 0
            t[0] = tj
        if tj > 1:
            t[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = r[tuple(t)]
            t[j] += 1
        for k in range(start, m):
            if t[k]:
                t[k] -= 1
                v += r[tuple(t)]
                t[k] += 1
        t[0] -= 1
        r[tuple(t)] = v * tj // (n - t[0])
    return r