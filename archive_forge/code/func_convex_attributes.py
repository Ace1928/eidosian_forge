import numpy as np
import scipy.sparse as sp
from cvxpy.atoms import diag, reshape
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
def convex_attributes(variables):
    """Returns a list of the (constraint-generating) convex attributes present
       among the variables.
    """
    return attributes_present(variables, CONVEX_ATTRIBUTES)