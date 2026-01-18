import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
def check_lib_lookup(qout, qin):
    status = True
    while status:
        try:
            action = qin.get()
        except Exception as e:
            qout.put(e)
            status = False
        else:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', NumbaWarning)
                    status, result = action()
                qout.put(result + (w,))
            except Exception as e:
                qout.put(e)
                status = False