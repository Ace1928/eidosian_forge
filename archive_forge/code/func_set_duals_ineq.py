from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def set_duals_ineq(self, duals_ineq):
    self._invalidate_duals_cache()
    np.copyto(self._duals_ineq, duals_ineq)
    self._duals_full[self._con_full_ineq_mask] = self._duals_ineq