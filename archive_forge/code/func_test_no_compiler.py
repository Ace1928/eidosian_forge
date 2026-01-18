import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_no_compiler(self):
    import distutils._msvccompiler as _msvccompiler

    def _find_vcvarsall(plat_spec):
        return (None, None)
    old_find_vcvarsall = _msvccompiler._find_vcvarsall
    _msvccompiler._find_vcvarsall = _find_vcvarsall
    try:
        self.assertRaises(DistutilsPlatformError, _msvccompiler._get_vc_env, 'wont find this version')
    finally:
        _msvccompiler._find_vcvarsall = old_find_vcvarsall