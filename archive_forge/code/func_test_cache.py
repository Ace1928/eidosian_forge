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
def test_cache(self):
    gpus = get_different_cc_gpus()
    if not gpus:
        self.skipTest('Need two different CCs for multi-CC cache test')
    self.check_pycache(0)
    mod = self.import_module()
    self.check_pycache(0)
    with gpus[0]:
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)
        self.check_hits(f.func, 0, 2)
        f = mod.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(6)
        self.check_hits(f.func, 0, 2)
    with gpus[1]:
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(6)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(6)
        self.check_hits(f.func, 0, 2)
        f = mod.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(6)
        self.check_hits(f.func, 0, 2)
    mod2 = self.import_module()
    self.assertIsNot(mod, mod2)
    with gpus[1]:
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(7)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(8)
        self.check_hits(f.func, 0, 2)
        f = mod2.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod2.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(10)
        self.check_hits(f.func, 0, 2)
    mod3 = self.import_module()
    self.assertIsNot(mod, mod3)
    with gpus[1]:
        f = mod3.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        f = mod3.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod3.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
    with gpus[0]:
        f = mod3.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        f = mod3.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod3.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))