import logging
import random
from pyomo.core import Var
def rand_guess_and_bound(val, lb, ub):
    """Random choice between current value and farthest bound."""
    far_bound = ub if ub - val >= val - lb else lb
    return random.uniform(val, far_bound)