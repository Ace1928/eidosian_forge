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
def dims_to_solver_dict(cone_dims):
    cones = dims_to_solver_dict_default(cone_dims)
    import scs
    if Version(scs.__version__) >= Version('3.0.0'):
        cones['z'] = cones.pop('f')
    return cones