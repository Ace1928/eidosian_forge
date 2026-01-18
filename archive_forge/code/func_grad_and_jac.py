import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import (
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import (CanonicalConstraint,
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport
def grad_and_jac(x):
    g = objective.grad(x)
    J_eq, _ = canonical.jac(x)
    return (g, J_eq)