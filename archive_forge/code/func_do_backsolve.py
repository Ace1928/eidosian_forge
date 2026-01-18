from pyomo.common.fileutils import find_library
from pyomo.contrib.pynumero.linalg.utils import validate_index, validate_value, _NotSet
import numpy.ctypeslib as npct
import numpy as np
import ctypes
import os
def do_backsolve(self, rhs, copy=True):
    rhs = rhs.astype(np.double, casting='safe', copy=copy)
    shape = rhs.shape
    if len(shape) == 1:
        rhs_dim = rhs.size
        nrhs = 1
        rhs = np.array([rhs])
    elif len(shape) == 2:
        raise NotImplementedError('Functionality for solving a matrix of right hand is buggy and needs fixing.')
        rhs_dim = rhs.shape[0]
        nrhs = rhs.shape[1]
    else:
        raise ValueError('Right hand side must be a one or two-dimensional array')
    assert self.dim_cached == rhs_dim, 'Dimension mismatch in RHS'
    if nrhs > 1:
        self.lib.set_nrhs(self._ma57, nrhs)
    if self.work_factor is not None:
        self.lib.alloc_work(self._ma57, int(self.work_factor * nrhs * rhs_dim))
    self.lib.do_backsolve(self._ma57, rhs_dim, rhs)
    if len(shape) == 1:
        rhs = rhs[0, :]
    return rhs