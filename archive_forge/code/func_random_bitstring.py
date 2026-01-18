from sympy.core import Basic, Integer
import random
def random_bitstring(n):
    """
    Generates a random bitlist of length n.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import random_bitstring
    >>> random_bitstring(3) # doctest: +SKIP
    100
    """
    return ''.join([random.choice('01') for i in range(n)])