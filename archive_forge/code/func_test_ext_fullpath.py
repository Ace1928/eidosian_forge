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
def test_ext_fullpath(self):
    ext = sysconfig.get_config_var('EXT_SUFFIX')
    dist = Distribution()
    cmd = self.build_ext(dist)
    cmd.inplace = 1
    cmd.distribution.package_dir = {'': 'src'}
    cmd.distribution.packages = ['lxml', 'lxml.html']
    curdir = os.getcwd()
    wanted = os.path.join(curdir, 'src', 'lxml', 'etree' + ext)
    path = cmd.get_ext_fullpath('lxml.etree')
    self.assertEqual(wanted, path)
    cmd.inplace = 0
    cmd.build_lib = os.path.join(curdir, 'tmpdir')
    wanted = os.path.join(curdir, 'tmpdir', 'lxml', 'etree' + ext)
    path = cmd.get_ext_fullpath('lxml.etree')
    self.assertEqual(wanted, path)
    build_py = cmd.get_finalized_command('build_py')
    build_py.package_dir = {}
    cmd.distribution.packages = ['twisted', 'twisted.runner.portmap']
    path = cmd.get_ext_fullpath('twisted.runner.portmap')
    wanted = os.path.join(curdir, 'tmpdir', 'twisted', 'runner', 'portmap' + ext)
    self.assertEqual(wanted, path)
    cmd.inplace = 1
    path = cmd.get_ext_fullpath('twisted.runner.portmap')
    wanted = os.path.join(curdir, 'twisted', 'runner', 'portmap' + ext)
    self.assertEqual(wanted, path)