from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def evaluate_hessian_lag(self, out=None):
    if not self._hessian_lag_is_cached:
        self._evaluate_objective_and_cache_if_necessary()
        self._evaluate_constraints_and_cache_if_necessary()
        data = np.zeros(self._nnz_hess_lag_lower, np.float64)
        self._asl.eval_hes_lag(self._primals, self._duals_full, data, obj_factor=self._obj_factor)
        values = np.concatenate((data, data[self._lower_hess_mask]))
        np.copyto(self._cached_hessian_lag.data, values)
        self._hessian_lag_is_cached = True
    if out is not None:
        if not isinstance(out, coo_matrix) or out.shape[0] != self._n_primals or out.shape[1] != self._n_primals or (out.nnz != self._nnz_hessian_lag):
            raise RuntimeError('evaluate_hessian_lag called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_primals, self._n_primals, self._nnz_hessian_lag))
        np.copyto(out.data, self._cached_hessian_lag.data)
        return out
    else:
        return self._cached_hessian_lag.copy()