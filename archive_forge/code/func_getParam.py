import os
import sys
import ctypes
import subprocess
import warnings
from uuid import uuid4
from .core import sparse, ctypesArrayFill, PulpSolverError
from .core import clock, log
from .core import LpSolver, LpSolver_CMD
from ..constants import (
from ..constants import LpContinuous, LpBinary, LpInteger
from ..constants import LpConstraintEQ, LpConstraintLE, LpConstraintGE
from ..constants import LpMinimize, LpMaximize
def getParam(self, name):
    """
            Get current value of parameter
            """
    par_dblval = ctypes.c_double()
    par_intval = ctypes.c_int()
    par_type = ctypes.c_int()
    par_name = coptstr(name)
    rc = self.SearchParamAttr(self.coptprob, par_name, byref(par_type))
    if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))
    if par_type.value == 0:
        rc = self.GetDblParam(self.coptprob, par_name, byref(par_dblval))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to get double parameter '{}'".format(par_name))
        else:
            retval = par_dblval.value
    elif par_type.value == 1:
        rc = self.GetIntParam(self.coptprob, par_name, byref(par_intval))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to get integer parameter '{}'".format(par_name))
        else:
            retval = par_intval.value
    else:
        raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))
    return retval