import difflib
import glob
import gzip
import os
import sys
import tempfile
import unittest
import Cython.Build.Dependencies
import Cython.Utils
from Cython.TestUtils import CythonTest
def test_multi_file_output(self):
    a_pyx = os.path.join(self.src_dir, 'a.pyx')
    a_c = a_pyx[:-4] + '.c'
    a_h = a_pyx[:-4] + '.h'
    a_api_h = a_pyx[:-4] + '_api.h'
    with open(a_pyx, 'w') as f:
        f.write('cdef public api int foo(int x): return x\n')
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    expected = [a_c, a_h, a_api_h]
    for output in expected:
        self.assertTrue(os.path.exists(output), output)
        os.unlink(output)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    for output in expected:
        self.assertTrue(os.path.exists(output), output)