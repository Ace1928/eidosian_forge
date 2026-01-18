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
@staticmethod
def get_spacing_matrix(shape: Tuple[int, ...], spacing, streak, num_blocks, offset):
    """Returns a sparse matrix that spaces out an expression.

        Parameters
        ----------
        shape : tuple
            (rows in matrix, columns in matrix)
        spacing : int
            The number of rows between the start of each non-zero block.
        streak: int
            The number of elements in each block.
        num_blocks : int
            The number of non-zero blocks.
        offset : int
            The number of zero rows at the beginning of the matrix.

        Returns
        -------
        SciPy CSC matrix
            A sparse matrix
        """
    num_values = num_blocks * streak
    val_arr = np.ones(num_values, dtype=np.float64)
    streak_plus_spacing = streak + spacing
    row_arr = np.arange(0, num_blocks * streak_plus_spacing).reshape(num_blocks, streak_plus_spacing)[:, :streak].flatten() + offset
    col_arr = np.arange(num_values)
    return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)