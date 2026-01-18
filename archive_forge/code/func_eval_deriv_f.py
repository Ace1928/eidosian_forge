from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def eval_deriv_f(self, x, df):
    assert x.size == self._nx, 'Error: Dimension mismatch.'
    assert x.dtype == np.double, 'Error: array type. Function eval_deriv_f expects an array of type double'
    res = self.ASLib.EXTERNAL_AmplInterface_eval_deriv_f(self._obj, x, df, len(x))
    if not res:
        raise PyNumeroEvaluationError('Error in AMPL evaluation')