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
def test_non_creatable_pycache(self):
    old_perms = os.stat(self.tempdir).st_mode
    os.chmod(self.tempdir, 320)
    self.addCleanup(os.chmod, self.tempdir, old_perms)
    self._test_pycache_fallback()