import os
import sys
import unittest
from copy import copy
from unittest import mock
from distutils.errors import DistutilsPlatformError, DistutilsByteCompileError
from distutils.util import (get_platform, convert_path, change_root,
from distutils import util # used to patch _environ_checked
from distutils.sysconfig import get_config_vars
from distutils import sysconfig
from distutils.tests import support
import _osx_support
def test_convert_path(self):
    os.sep = '/'

    def _join(path):
        return '/'.join(path)
    os.path.join = _join
    self.assertEqual(convert_path('/home/to/my/stuff'), '/home/to/my/stuff')
    os.sep = '\\'

    def _join(*path):
        return '\\'.join(path)
    os.path.join = _join
    self.assertRaises(ValueError, convert_path, '/home/to/my/stuff')
    self.assertRaises(ValueError, convert_path, 'home/to/my/stuff/')
    self.assertEqual(convert_path('home/to/my/stuff'), 'home\\to\\my\\stuff')
    self.assertEqual(convert_path('.'), os.curdir)