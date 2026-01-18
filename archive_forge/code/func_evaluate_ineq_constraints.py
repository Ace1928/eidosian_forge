from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def evaluate_ineq_constraints(self, out=None):
    self._evaluate_constraints_and_cache_if_necessary()
    if out is not None:
        if not isinstance(out, np.ndarray) or out.size != self._n_con_ineq:
            raise RuntimeError('Called evaluate_ineq_constraints with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_con_ineq))
        self._cached_con_full.compress(self._con_full_ineq_mask, out=out)
        return out
    else:
        return self._cached_con_full.compress(self._con_full_ineq_mask)