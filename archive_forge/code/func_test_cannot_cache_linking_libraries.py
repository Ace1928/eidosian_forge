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
def test_cannot_cache_linking_libraries(self):
    link = str(test_data_dir / 'jitlink.ptx')
    msg = 'Cannot pickle CUDACodeLibrary with linking files'
    with self.assertRaisesRegex(RuntimeError, msg):

        @cuda.jit('void()', cache=True, link=[link])
        def f():
            pass