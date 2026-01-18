import multiprocessing as mp
import logging
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_cuda_python,
from numba.tests.support import linux_only
def child_test_wrapper(result_queue):
    try:
        output = child_test()
        success = True
    except:
        output = traceback.format_exc()
        success = False
    result_queue.put((success, output))