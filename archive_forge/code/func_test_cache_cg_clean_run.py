import multiprocessing
import os
import shutil
import subprocess
import sys
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
@skip_unless_cc_60
@skip_if_cudadevrt_missing
@skip_if_mvc_enabled('CG not supported with MVC')
def test_cache_cg_clean_run(self):
    self.check_pycache(0)
    code = 'if 1:\n            import sys\n\n            sys.path.insert(0, %(tempdir)r)\n            mod = __import__(%(modname)r)\n            mod.cg_usecase(0)\n            ' % dict(tempdir=self.tempdir, modname=self.modname)
    popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = popen.communicate(timeout=60)
    if popen.returncode != 0:
        raise AssertionError('process failed with code %s: \nstdout follows\n%s\nstderr follows\n%s\n' % (popen.returncode, out.decode(), err.decode()))