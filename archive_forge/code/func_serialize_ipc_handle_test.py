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
def serialize_ipc_handle_test(handle, result_queue):

    def the_work():
        dtype = np.dtype(np.intp)
        darr = handle.open_array(cuda.current_context(), shape=handle.size // dtype.itemsize, dtype=dtype)
        arr = darr.copy_to_host()
        handle.close()
        return arr
    core_ipc_handle_test(the_work, result_queue)