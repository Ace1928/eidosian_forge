import random
import warnings
from itertools import repeat
def cxUniformPartialyMatched(ind1, ind2, indpb):
    """Executes a uniform partially matched crossover (UPMX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values chosen at random with a probability of *indpb* in the two
    parents and swapping the values of those indexes. For more details see
    [Cicirello2000]_.

    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.

    .. [Cicirello2000] Cicirello and Smith, "Modeling GA performance for
       control parameter optimization", 2000.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = ([0] * size, [0] * size)
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    for i in range(size):
        if random.random() < indpb:
            temp1 = ind1[i]
            temp2 = ind2[i]
            ind1[i], ind1[p1[temp2]] = (temp2, temp1)
            ind2[i], ind2[p2[temp1]] = (temp1, temp2)
            p1[temp1], p1[temp2] = (p1[temp2], p1[temp1])
            p2[temp1], p2[temp2] = (p2[temp2], p2[temp1])
    return (ind1, ind2)