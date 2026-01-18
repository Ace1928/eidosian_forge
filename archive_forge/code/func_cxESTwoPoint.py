import random
import warnings
from itertools import repeat
def cxESTwoPoint(ind1, ind2):
    """Executes a classical two points crossover on both the individuals and their
    strategy. The individuals shall be a :term:`sequence` and must have a
    :term:`sequence` :attr:`strategy` attribute. The crossover points for the
    individual and the strategy are the same.

    :param ind1: The first evolution strategy participating in the crossover.
    :param ind2: The second evolution strategy participating in the crossover.
    :returns: A tuple of two evolution strategies.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    pt1 = random.randint(1, size)
    pt2 = random.randint(1, size - 1)
    if pt2 >= pt1:
        pt2 += 1
    else:
        pt1, pt2 = (pt2, pt1)
    ind1[pt1:pt2], ind2[pt1:pt2] = (ind2[pt1:pt2], ind1[pt1:pt2])
    ind1.strategy[pt1:pt2], ind2.strategy[pt1:pt2] = (ind2.strategy[pt1:pt2], ind1.strategy[pt1:pt2])
    return (ind1, ind2)