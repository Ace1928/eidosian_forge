import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def base_ipc_handle_test(handle, size, result_queue):

    def the_work():
        dtype = np.dtype(np.intp)
        with cuda.open_ipc_array(handle, shape=size // dtype.itemsize, dtype=dtype) as darr:
            return darr.copy_to_host()
    core_ipc_handle_test(the_work, result_queue)