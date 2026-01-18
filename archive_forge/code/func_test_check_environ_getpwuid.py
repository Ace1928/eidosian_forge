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
@unittest.skipUnless(os.name == 'posix', 'specific to posix')
def test_check_environ_getpwuid(self):
    util._environ_checked = 0
    os.environ.pop('HOME', None)
    try:
        import pwd
    except ImportError:
        raise unittest.SkipTest('Test requires pwd module.')
    result = pwd.struct_passwd((None, None, None, None, None, '/home/distutils', None))
    with mock.patch.object(pwd, 'getpwuid', return_value=result):
        check_environ()
        self.assertEqual(os.environ['HOME'], '/home/distutils')
    util._environ_checked = 0
    os.environ.pop('HOME', None)
    with mock.patch.object(pwd, 'getpwuid', side_effect=KeyError):
        check_environ()
        self.assertNotIn('HOME', os.environ)