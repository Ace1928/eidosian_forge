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
def test_ipc_handle(self):
    arr = np.arange(10, dtype=np.intp)
    devarr = cuda.to_device(arr)
    ctx = cuda.current_context()
    ipch = ctx.get_ipc_handle(devarr.gpu_data)
    if driver.USE_NV_BINDING:
        handle_bytes = ipch.handle.reserved
    else:
        handle_bytes = bytes(ipch.handle)
    size = ipch.size
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    args = (handle_bytes, size, result_queue)
    proc = ctx.Process(target=base_ipc_handle_test, args=args)
    proc.start()
    succ, out = result_queue.get()
    if not succ:
        self.fail(out)
    else:
        np.testing.assert_equal(arr, out)
    proc.join(3)