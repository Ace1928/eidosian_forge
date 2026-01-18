import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.utilities.versioning import Version
@staticmethod
def extract_dual_value(result_vec, offset, constraint):
    """Extracts the dual value for constraint starting at offset.

        Special cases PSD constraints, as per the SCS specification.
        """
    if isinstance(constraint, PSD):
        dim = constraint.shape[0]
        lower_tri_dim = dim * (dim + 1) // 2
        new_offset = offset + lower_tri_dim
        lower_tri = result_vec[offset:new_offset]
        full = tri_to_full(lower_tri, dim)
        return (full, new_offset)
    else:
        return utilities.extract_dual_value(result_vec, offset, constraint)