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
def test_cannot_serialize_unfinalized(self):
    from numba.cuda.codegen import CUDACodeLibrary
    codegen = object()
    name = 'library'
    cl = CUDACodeLibrary(codegen, name)
    with self.assertRaisesRegex(RuntimeError, 'Cannot pickle unfinalized'):
        cl._reduce_states()