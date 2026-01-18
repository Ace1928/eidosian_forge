import sys
import os
from io import StringIO
import textwrap
from distutils.core import Distribution
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.tests.support import (TempdirManager, LoggingSilencer,
from distutils.extension import Extension
from distutils.errors import (
import unittest
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok
from test.support import threading_helper
@unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for MacOSX')
def test_deployment_target_higher_ok(self):
    deptarget = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
    if deptarget:
        deptarget = [int(x) for x in deptarget.split('.')]
        deptarget[-1] += 1
        deptarget = '.'.join((str(i) for i in deptarget))
        self._try_compile_deployment_target('<', deptarget)