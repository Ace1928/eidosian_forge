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
def test_grok_environment_error(self):
    exc = IOError('Unable to find batch file')
    msg = grok_environment_error(exc)
    self.assertEqual(msg, 'error: Unable to find batch file')