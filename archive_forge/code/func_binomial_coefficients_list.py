from sympy.utilities.misc import as_int
def binomial_coefficients_list(n):
    """ Return a list of binomial coefficients as rows of the Pascal's
    triangle.

    Examples
    ========

    >>> from sympy.ntheory import binomial_coefficients_list
    >>> binomial_coefficients_list(9)
    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]

    See Also
    ========

    binomial_coefficients, multinomial_coefficients
    """
    n = as_int(n)
    d = [1] * (n + 1)
    a = 1
    for k in range(1, n // 2 + 1):
        a = a * (n - k + 1) // k
        d[k] = d[n - k] = a
    return d