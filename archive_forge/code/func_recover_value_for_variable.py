import numpy as np
import scipy.sparse as sp
from cvxpy.atoms import diag, reshape
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
def recover_value_for_variable(variable, lowered_value, project: bool=True):
    if variable.attributes['diag']:
        return sp.diags(lowered_value.flatten())
    elif attributes_present([variable], SYMMETRIC_ATTRIBUTES):
        n = variable.shape[0]
        value = np.zeros(variable.shape)
        idxs = np.triu_indices(n)
        value[idxs] = lowered_value.flatten()
        return value + value.T - np.diag(value.diagonal())
    elif project:
        return variable.project(lowered_value)
    else:
        return lowered_value