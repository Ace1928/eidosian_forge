import multiprocessing as mp
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
from numba.tests.support import linux_only
def test_mvc(self):
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(target=child_test_wrapper, args=(result_queue,))
    proc.start()
    proc.join()
    success, output = result_queue.get()
    if not success:
        self.fail(output)