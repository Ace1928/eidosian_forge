import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_get_vc2017(self):
    import distutils._msvccompiler as _msvccompiler
    version, path = _msvccompiler._find_vc2017()
    if version:
        self.assertGreaterEqual(version, 15)
        self.assertTrue(os.path.isdir(path))
    else:
        raise unittest.SkipTest('VS 2017 is not installed')