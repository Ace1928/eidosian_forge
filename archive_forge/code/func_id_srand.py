import scipy.linalg._interpolative as _id
import numpy as np
def id_srand(n):
    """
    Generate standard uniform pseudorandom numbers via a very efficient lagged
    Fibonacci method.

    :param n:
        Number of pseudorandom numbers to generate.
    :type n: int

    :return:
        Pseudorandom numbers.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.id_srand(n)