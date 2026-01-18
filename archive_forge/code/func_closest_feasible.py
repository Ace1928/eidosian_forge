from functools import wraps
from itertools import repeat
def closest_feasible(individual):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = numpy.array(individual)
    feasible_ind = numpy.maximum(MIN_BOUND, feasible_ind)
    feasible_ind = numpy.minimum(MAX_BOUND, feasible_ind)
    return feasible_ind