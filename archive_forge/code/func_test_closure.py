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
def test_closure(self):
    mod = self.import_module()
    with warnings.catch_warnings():
        warnings.simplefilter('error', NumbaWarning)
        f = mod.closure1
        self.assertPreciseEqual(f(3), 6)
        f = mod.closure2
        self.assertPreciseEqual(f(3), 8)
        f = mod.closure3
        self.assertPreciseEqual(f(3), 10)
        f = mod.closure4
        self.assertPreciseEqual(f(3), 12)
        self.check_pycache(5)