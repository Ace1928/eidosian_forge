from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
def increase_memory_allocation(self, factor):
    if self._prev_allocation == 0:
        new_allocation = 1
    else:
        new_allocation = int(factor * self._prev_allocation)
    self.set_icntl(23, new_allocation)
    self._prev_allocation = new_allocation
    return new_allocation