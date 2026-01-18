from sympy.utilities.misc import as_int
def binomial_coefficients(n):
    """Return a dictionary containing pairs :math:`{(k1,k2) : C_kn}` where
    :math:`C_kn` are binomial coefficients and :math:`n=k1+k2`.

    Examples
    ========

    >>> from sympy.ntheory import binomial_coefficients
    >>> binomial_coefficients(9)
    {(0, 9): 1, (1, 8): 9, (2, 7): 36, (3, 6): 84,
     (4, 5): 126, (5, 4): 126, (6, 3): 84, (7, 2): 36, (8, 1): 9, (9, 0): 1}

    See Also
    ========

    binomial_coefficients_list, multinomial_coefficients
    """
    n = as_int(n)
    d = {(0, n): 1, (n, 0): 1}
    a = 1
    for k in range(1, n // 2 + 1):
        a = a * (n - k + 1) // k
        d[k, n - k] = d[n - k, k] = a
    return d