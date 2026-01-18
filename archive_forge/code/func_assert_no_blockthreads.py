import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
def assert_no_blockthreads():
    blockthreads = []
    for t in threading.enumerate():
        if not isinstance(t, simulator.kernel.BlockThread):
            continue
        t.join(1)
        if t.is_alive():
            self.fail('Blocked kernel thread: %s' % t)
    self.assertListEqual(blockthreads, [])