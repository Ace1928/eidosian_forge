from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def reset_info(self, info):
    """
        Reset information after problem updates
        """
    info.solve_time = 0.0
    info.polish_time = 0.0
    self.update_status(OSQP_UNSOLVED)
    info.rho_updates = 0