import re
def quadruples_with_fixed_sum_iterator(N, skipVertices=False):
    """
    Similar to tuples_with_fixed_sum_iterator for quadruples.

    >>> list(quadruples_with_fixed_sum_iterator(2, skipVertices = True))
    [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)]
    """
    return tuples_with_fixed_sum_iterator(N, 4, skipVertices=skipVertices)