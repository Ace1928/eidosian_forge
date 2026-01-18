from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def eval_g(self, x, g):
    assert x.size == self._nx, 'Error: Dimension mismatch.'
    assert g.size == self._ny, 'Error: Dimension mismatch.'
    assert x.dtype == np.double, 'Error: array type. Function eval_g expects an array of type double'
    assert g.dtype == np.double, 'Error: array type. Function eval_g expects an array of type double'
    res = self.ASLib.EXTERNAL_AmplInterface_eval_g(self._obj, x, self._nx, g, self._ny)
    if not res:
        raise PyNumeroEvaluationError('Error in AMPL evaluation')