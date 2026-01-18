from sympy.utilities.misc import as_int
def multinomial_coefficients_iterator(m, n, _tuple=tuple):
    """multinomial coefficient iterator

    This routine has been optimized for `m` large with respect to `n` by taking
    advantage of the fact that when the monomial tuples `t` are stripped of
    zeros, their coefficient is the same as that of the monomial tuples from
    ``multinomial_coefficients(n, n)``. Therefore, the latter coefficients are
    precomputed to save memory and time.

    >>> from sympy.ntheory.multinomial import multinomial_coefficients
    >>> m53, m33 = multinomial_coefficients(5,3), multinomial_coefficients(3,3)
    >>> m53[(0,0,0,1,2)] == m53[(0,0,1,0,2)] == m53[(1,0,2,0,0)] == m33[(0,1,2)]
    True

    Examples
    ========

    >>> from sympy.ntheory.multinomial import multinomial_coefficients_iterator
    >>> it = multinomial_coefficients_iterator(20,3)
    >>> next(it)
    ((3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1)
    """
    m = as_int(m)
    n = as_int(n)
    if m < 2 * n or n == 1:
        mc = multinomial_coefficients(m, n)
        yield from mc.items()
    else:
        mc = multinomial_coefficients(n, n)
        mc1 = {}
        for k, v in mc.items():
            mc1[_tuple(filter(None, k))] = v
        mc = mc1
        t = [n] + [0] * (m - 1)
        t1 = _tuple(t)
        b = _tuple(filter(None, t1))
        yield (t1, mc[b])
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
            else:
                j += 1
                t[j] += 1
            t[0] -= 1
            t1 = _tuple(t)
            b = _tuple(filter(None, t1))
            yield (t1, mc[b])