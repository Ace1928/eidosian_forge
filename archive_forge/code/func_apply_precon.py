import numpy as np
from ase.optimize.sciopt import SciPyOptimizer, OptimizerConvergenceError
def apply_precon(F, X):
    return (F, residual(F, X))