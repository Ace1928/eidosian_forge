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
@skip_bad_access
@unittest.skipIf(os.name == 'nt', 'cannot easily make a directory read-only on Windows')
def test_non_writable_pycache(self):
    pycache = os.path.join(self.tempdir, '__pycache__')
    os.mkdir(pycache)
    old_perms = os.stat(pycache).st_mode
    os.chmod(pycache, 320)
    self.addCleanup(os.chmod, pycache, old_perms)
    self._test_pycache_fallback()