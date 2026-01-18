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
def split_adjoint(self, del_vars=None):
    """Adjoint of split_solution.
        """
    raise NotImplementedError