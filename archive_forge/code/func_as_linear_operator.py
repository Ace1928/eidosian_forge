from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver
def as_linear_operator(linear_op):
    if isinstance(linear_op, LinearOperator):
        return linear_op
    elif sp.issparse(linear_op):
        return LinearOperator(linear_op, linear_op.shape)