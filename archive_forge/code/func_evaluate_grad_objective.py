from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def evaluate_grad_objective(self, out=None):
    if not self._grad_objective_is_cached:
        self._asl.eval_deriv_f(self._primals, self._cached_grad_objective)
        self._grad_objective_is_cached = True
    if out is not None:
        if not isinstance(out, np.ndarray) or out.size != self._n_primals:
            raise RuntimeError('Called evaluate_grad_objective with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_primals))
        np.copyto(out, self._cached_grad_objective)
        return out
    else:
        return self._cached_grad_objective.copy()