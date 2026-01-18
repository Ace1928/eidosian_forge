from pyomo.common import unittest
from pyomo.contrib.pynumero.dependencies import numpy_available, scipy_available
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from scipy.sparse import coo_matrix, spmatrix
import numpy as np
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.linalg.ma57_interface import MA57
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU, ScipyIterative
from scipy.sparse.linalg import gmres
from pyomo.contrib.pynumero.linalg.mumps_interface import (
def solve_helper(self, m: np.ndarray, x: np.ndarray, solver: LinearSolverInterface):
    bm, bx, br = self.create_blocks(m, x)
    bx2, res = solver.solve(bm, br)
    self.assertEqual(res.status, LinearSolverStatus.successful)
    err = np.max(np.abs(bx - bx2))
    self.assertAlmostEqual(err, 0)