from .base_linear_solver_interface import IPLinearSolverInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus, LinearSolverResults
from pyomo.common.dependencies import attempt_import
from collections import OrderedDict
from typing import Union, Optional, Tuple
from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np
from pyomo.contrib.pynumero.linalg.mumps_interface import (
def get_inertia(self):
    num_negative_eigenvalues = self.get_infog(12)
    num_zero_eigenvalues = self.get_infog(28)
    num_positive_eigenvalues = self._dim - num_negative_eigenvalues - num_zero_eigenvalues
    return (num_positive_eigenvalues, num_negative_eigenvalues, num_zero_eigenvalues)