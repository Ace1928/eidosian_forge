from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def eval_hes_lag(self, x, lam, hes_lag, obj_factor=1.0):
    assert x.size == self._nx, 'Error: Dimension mismatch.'
    assert lam.size == self._ny, 'Error: Dimension mismatch.'
    assert hes_lag.size == self._nnz_hess, 'Error: Dimension mismatch.'
    assert x.dtype == np.double, 'Error: array type. Function eval_hes_lag expects an array of type double'
    assert lam.dtype == np.double, 'Error: array type. Function eval_hes_lag expects an array of type double'
    assert hes_lag.dtype == np.double, 'Error: array type. Function eval_hes_lag expects an array of type double'
    if self.interface_version >= 1:
        res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(self._obj, x, self._nx, lam, self._ny, hes_lag, self._nnz_hess, obj_factor)
    else:
        res = self.ASLib.EXTERNAL_AmplInterface_eval_hes_lag(self._obj, x, self._nx, lam, self._ny, hes_lag, self._nnz_hess)
    if not res:
        raise PyNumeroEvaluationError('Error in AMPL evaluation')