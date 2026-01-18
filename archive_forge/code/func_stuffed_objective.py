from __future__ import annotations
import numpy as np
import cvxpy.settings as s
from cvxpy.constraints import (
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import nonpos2nonneg
from cvxpy.reductions.matrix_stuffing import MatrixStuffing, extract_mip_idx
from cvxpy.reductions.utilities import (
from cvxpy.utilities.coeff_extractor import CoeffExtractor
def stuffed_objective(self, problem, extractor):
    expr = problem.objective.expr.copy()
    params_to_P, params_to_q = extractor.quad_form(expr)
    params_to_P = 2 * params_to_P
    boolean, integer = extract_mip_idx(problem.variables())
    x = Variable(extractor.x_length, boolean=boolean, integer=integer)
    return (params_to_P, params_to_q, x)