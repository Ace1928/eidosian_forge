from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def struct_hes_lag(self, irow, jcol):
    irow_p = irow.astype(np.intc, casting='safe', copy=False)
    jcol_p = jcol.astype(np.intc, casting='safe', copy=False)
    assert len(irow) == len(jcol), 'Error: Dimension mismatch. Arrays irow and jcol must be of the same size'
    assert len(irow) == self._nnz_hess, 'Error: Dimension mismatch. Hessian has {} nnz'.format(self._nnz_hess)
    self.ASLib.EXTERNAL_AmplInterface_struct_hes_lag(self._obj, irow_p, jcol_p, len(irow))