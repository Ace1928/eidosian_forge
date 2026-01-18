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
def test_cycache_uses_cache(self):
    a_pyx = os.path.join(self.src_dir, 'a.pyx')
    a_c = a_pyx[:-4] + '.c'
    with open(a_pyx, 'w') as f:
        f.write('pass')
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    a_cache = os.path.join(self.cache_dir, os.listdir(self.cache_dir)[0])
    with gzip.GzipFile(a_cache, 'wb') as gzipfile:
        gzipfile.write('fake stuff'.encode('ascii'))
    os.unlink(a_c)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    with open(a_c) as f:
        a_contents = f.read()
    self.assertEqual(a_contents, 'fake stuff', 'Unexpected contents: %s...' % a_contents[:100])