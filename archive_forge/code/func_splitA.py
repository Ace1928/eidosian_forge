import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def splitA(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = ([], [])
    best_b, worst_b = ([], [])
    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)
    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))
    if balance_a <= balance_b:
        return (best_a, worst_a)
    else:
        return (best_b, worst_b)