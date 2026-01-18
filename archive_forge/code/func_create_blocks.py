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
def create_blocks(self, m: np.ndarray, x: np.ndarray):
    m = coo_matrix(m)
    r = m * x
    bm = BlockMatrix(2, 2)
    bm.set_block(0, 0, m.copy())
    bm.set_block(1, 1, m.copy())
    br = BlockVector(2)
    br.set_block(0, r.copy())
    br.set_block(1, r.copy())
    bx = BlockVector(2)
    bx.set_block(0, x.copy())
    bx.set_block(1, x.copy())
    return (bm, bx, br)