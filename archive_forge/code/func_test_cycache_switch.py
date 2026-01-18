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
def test_cycache_switch(self):
    content1 = 'value = 1\n'
    content2 = 'value = 2\n'
    a_pyx = os.path.join(self.src_dir, 'a.pyx')
    a_c = a_pyx[:-4] + '.c'
    with open(a_pyx, 'w') as f:
        f.write(content1)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    self.assertEqual(1, len(self.cache_files('a.c*')))
    with open(a_c) as f:
        a_contents1 = f.read()
    os.unlink(a_c)
    with open(a_pyx, 'w') as f:
        f.write(content2)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    with open(a_c) as f:
        a_contents2 = f.read()
    os.unlink(a_c)
    self.assertNotEqual(a_contents1, a_contents2, 'C file not changed!')
    self.assertEqual(2, len(self.cache_files('a.c*')))
    with open(a_pyx, 'w') as f:
        f.write(content1)
    self.fresh_cythonize(a_pyx, cache=self.cache_dir)
    self.assertEqual(2, len(self.cache_files('a.c*')))
    with open(a_c) as f:
        a_contents = f.read()
    self.assertEqual(a_contents, a_contents1, msg='\n'.join(list(difflib.unified_diff(a_contents.split('\n'), a_contents1.split('\n')))[:10]))