import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
@unittest.skipUnless(has_gitpython, 'Requires gitpython')
def test_gitdiff(self):
    try:
        subprocess.call('git', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        self.skipTest('no git available')
    outs = self.get_testsuite_listing(['-g'])
    self.assertNotIn('Git diff by common ancestor', outs)
    outs = self.get_testsuite_listing(['-g=ancestor'])
    self.assertIn('Git diff by common ancestor', outs)
    subp_kwargs = dict(stderr=subprocess.DEVNULL)
    with self.assertRaises(subprocess.CalledProcessError):
        self.get_testsuite_listing(['-g=ancest'], subp_kwargs=subp_kwargs)