import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def sweepA(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair] + 1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)