from sympy.core import Integer, Pow, Mod
from sympy import factorint
def is_abelian_number(n):
    """
    Check whether `n` is an abelian number. A number `n` is said to be abelian
    if and only if every finite group of order `n` is abelian. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_abelian_number
    >>> from sympy import randprime
    >>> is_abelian_number(4)
    True
    >>> is_abelian_number(randprime(1, 2000)**2)
    True
    >>> is_abelian_number(60)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., *Nilpotent Numbers*,
            The American Mathematical Monthly, 107(7), 631-634.


    """
    if n <= 0 or int(n) != n:
        raise ValueError('n must be a positive integer, not %i' % n)
    n = Integer(n)
    if not is_nilpotent_number(n):
        return False
    prime_factors = list(factorint(n).items())
    is_abelian = all((a_i < 3 for p_i, a_i in prime_factors))
    return is_abelian