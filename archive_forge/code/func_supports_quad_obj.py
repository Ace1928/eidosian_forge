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
def supports_quad_obj(self) -> bool:
    """SCS >= 3.0.0 supports a quadratic objective.
        """
    import scs
    return Version(scs.__version__) >= Version('3.0.0')