import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """Finds the individuals with extreme values for each objective function."""
    if extreme_points is not None:
        fitnesses = numpy.concatenate((fitnesses, extreme_points), axis=0)
    ft = fitnesses - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1000000.0
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)
    min_asf_idx = numpy.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]