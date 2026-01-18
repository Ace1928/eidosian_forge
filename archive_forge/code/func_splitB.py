import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def splitB(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = ([], [], [], [])
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)
    worst1_a, worst2_a, worst1_b, worst2_b = ([], [], [], [])
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)
    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))
    if balance_a <= balance_b:
        return (best1_a, best2_a, worst1_a, worst2_a)
    else:
        return (best1_b, best2_b, worst1_b, worst2_b)