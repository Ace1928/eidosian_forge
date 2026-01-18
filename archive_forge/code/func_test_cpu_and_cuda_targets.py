import multiprocessing
import os
import shutil
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
def test_cpu_and_cuda_targets(self):
    self.check_pycache(0)
    mod = self.import_module()
    self.check_pycache(0)
    f_cpu = mod.assign_cpu
    f_cuda = mod.assign_cuda
    self.assertPreciseEqual(f_cpu(5), 5)
    self.check_pycache(2)
    self.assertPreciseEqual(f_cuda(5), 5)
    self.check_pycache(3)
    self.check_hits(f_cpu.func, 0, 1)
    self.check_hits(f_cuda.func, 0, 1)
    self.assertPreciseEqual(f_cpu(5.5), 5.5)
    self.check_pycache(4)
    self.assertPreciseEqual(f_cuda(5.5), 5.5)
    self.check_pycache(5)
    self.check_hits(f_cpu.func, 0, 2)
    self.check_hits(f_cuda.func, 0, 2)