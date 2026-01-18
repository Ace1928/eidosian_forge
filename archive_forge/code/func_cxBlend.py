import random
import warnings
from itertools import repeat
def cxBlend(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[i] = (1.0 - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1.0 - gamma) * x2
    return (ind1, ind2)