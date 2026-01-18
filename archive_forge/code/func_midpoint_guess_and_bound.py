import logging
import random
from pyomo.core import Var
def midpoint_guess_and_bound(val, lb, ub):
    """Midpoint between current value and farthest bound."""
    far_bound = ub if ub - val >= val - lb else lb
    return (far_bound + val) / 2