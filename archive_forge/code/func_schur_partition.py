import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer
def schur_partition(n):
    """

    This function returns the partition in the minimum number of sum-free subsets
    according to the lower bound given by the Schur Number.

    Parameters
    ==========

    n: a number
        n is the upper limit of the range [1, n] for which we need to find and
        return the minimum number of free subsets according to the lower bound
        of schur number

    Returns
    =======

    List of lists
        List of the minimum number of sum-free subsets

    Notes
    =====

    It is possible for some n to make the partition into less
    subsets since the only known Schur numbers are:
    S(1) = 1, S(2) = 4, S(3) = 13, S(4) = 44.
    e.g for n = 44 the lower bound from the function above is 5 subsets but it has been proven
    that can be done with 4 subsets.

    Examples
    ========

    For n = 1, 2, 3 the answer is the set itself

    >>> from sympy.combinatorics.schur_number import schur_partition
    >>> schur_partition(2)
    [[1, 2]]

    For n > 3, the answer is the minimum number of sum-free subsets:

    >>> schur_partition(5)
    [[3, 2], [5], [1, 4]]

    >>> schur_partition(8)
    [[3, 2], [6, 5, 8], [1, 4, 7]]
    """
    if isinstance(n, Basic) and (not n.is_Number):
        raise ValueError('Input value must be a number')
    number_of_subsets = _schur_subsets_number(n)
    if n == 1:
        sum_free_subsets = [[1]]
    elif n == 2:
        sum_free_subsets = [[1, 2]]
    elif n == 3:
        sum_free_subsets = [[1, 2, 3]]
    else:
        sum_free_subsets = [[1, 4], [2, 3]]
    while len(sum_free_subsets) < number_of_subsets:
        sum_free_subsets = _generate_next_list(sum_free_subsets, n)
        missed_elements = [3 * k + 1 for k in range(len(sum_free_subsets), (n - 1) // 3 + 1)]
        sum_free_subsets[-1] += missed_elements
    return sum_free_subsets