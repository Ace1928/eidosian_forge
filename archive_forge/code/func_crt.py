from rsa._compat import zip
def crt(a_values, modulo_values):
    """Chinese Remainder Theorem.

    Calculates x such that x = a[i] (mod m[i]) for each i.

    :param a_values: the a-values of the above equation
    :param modulo_values: the m-values of the above equation
    :returns: x such that x = a[i] (mod m[i]) for each i


    >>> crt([2, 3], [3, 5])
    8

    >>> crt([2, 3, 2], [3, 5, 7])
    23

    >>> crt([2, 3, 0], [7, 11, 15])
    135
    """
    m = 1
    x = 0
    for modulo in modulo_values:
        m *= modulo
    for m_i, a_i in zip(modulo_values, a_values):
        M_i = m // m_i
        inv = inverse(M_i, m_i)
        x = (x + a_i * M_i * inv) % m
    return x