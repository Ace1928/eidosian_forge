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
def check_ipc_handle_serialization(self, index_arg=None, foreign=False):
    arr = np.arange(10, dtype=np.intp)
    devarr = cuda.to_device(arr)
    if index_arg is not None:
        devarr = devarr[index_arg]
    if foreign:
        devarr = cuda.as_cuda_array(ForeignArray(devarr))
    expect = devarr.copy_to_host()
    ctx = cuda.current_context()
    ipch = ctx.get_ipc_handle(devarr.gpu_data)
    buf = pickle.dumps(ipch)
    ipch_recon = pickle.loads(buf)
    self.assertIs(ipch_recon.base, None)
    self.assertEqual(ipch_recon.size, ipch.size)
    if driver.USE_NV_BINDING:
        self.assertEqual(ipch_recon.handle.reserved, ipch.handle.reserved)
    else:
        self.assertEqual(tuple(ipch_recon.handle), tuple(ipch.handle))
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    args = (ipch, result_queue)
    proc = ctx.Process(target=serialize_ipc_handle_test, args=args)
    proc.start()
    succ, out = result_queue.get()
    if not succ:
        self.fail(out)
    else:
        np.testing.assert_equal(expect, out)
    proc.join(3)