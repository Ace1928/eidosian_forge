import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
def dims_to_solver_cones(cone_dims):
    import clarabel
    cones = []
    if cone_dims.zero > 0:
        cones.append(clarabel.ZeroConeT(cone_dims.zero))
    if cone_dims.nonneg > 0:
        cones.append(clarabel.NonnegativeConeT(cone_dims.nonneg))
    for dim in cone_dims.soc:
        cones.append(clarabel.SecondOrderConeT(dim))
    for dim in cone_dims.psd:
        cones.append(clarabel.PSDTriangleConeT(dim))
    for _ in range(cone_dims.exp):
        cones.append(clarabel.ExponentialConeT())
    for pow in cone_dims.p3d:
        cones.append(clarabel.PowerConeT(pow))
    return cones