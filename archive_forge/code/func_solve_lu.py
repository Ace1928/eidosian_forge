import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
from .base import OdeSolver, DenseOutput
def solve_lu(LU, b):
    return lu_solve(LU, b, overwrite_b=True)