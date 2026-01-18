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
def test_cache_reuse(self):
    mod = self.import_module()
    mod.add_usecase(2, 3)
    mod.add_usecase(2.5, 3.5)
    mod.outer_uncached(2, 3)
    mod.outer(2, 3)
    mod.record_return_packed(mod.packed_arr, 0)
    mod.record_return_aligned(mod.aligned_arr, 1)
    mod.simple_usecase_caller(2)
    mtimes = self.get_cache_mtimes()
    self.check_hits(mod.add_usecase.func, 0, 2)
    mod2 = self.import_module()
    self.assertIsNot(mod, mod2)
    f = mod2.add_usecase
    f(2, 3)
    self.check_hits(f.func, 1, 0)
    f(2.5, 3.5)
    self.check_hits(f.func, 2, 0)
    self.assertEqual(self.get_cache_mtimes(), mtimes)
    self.run_in_separate_process()
    self.assertEqual(self.get_cache_mtimes(), mtimes)