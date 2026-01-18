import numpy as np
import scipy.sparse as sp
from cvxpy.atoms import diag, reshape
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
def attributes_present(variables, attr_map):
    """Returns a list of the relevant attributes present
       among the variables.
    """
    return [attr for attr in attr_map if any((v.attributes[attr] for v in variables))]