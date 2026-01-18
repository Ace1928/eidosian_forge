from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
def get_icntl(self, key):
    return self._mumps.id.icntl[key - 1]