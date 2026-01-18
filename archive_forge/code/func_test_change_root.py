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
def test_change_root(self):
    os.name = 'posix'

    def _isabs(path):
        return path[0] == '/'
    os.path.isabs = _isabs

    def _join(*path):
        return '/'.join(path)
    os.path.join = _join
    self.assertEqual(change_root('/root', '/old/its/here'), '/root/old/its/here')
    self.assertEqual(change_root('/root', 'its/here'), '/root/its/here')
    os.name = 'nt'

    def _isabs(path):
        return path.startswith('c:\\')
    os.path.isabs = _isabs

    def _splitdrive(path):
        if path.startswith('c:'):
            return ('', path.replace('c:', ''))
        return ('', path)
    os.path.splitdrive = _splitdrive

    def _join(*path):
        return '\\'.join(path)
    os.path.join = _join
    self.assertEqual(change_root('c:\\root', 'c:\\old\\its\\here'), 'c:\\root\\old\\its\\here')
    self.assertEqual(change_root('c:\\root', 'its\\here'), 'c:\\root\\its\\here')
    os.name = 'BugsBunny'
    self.assertRaises(DistutilsPlatformError, change_root, 'c:\\root', 'its\\here')