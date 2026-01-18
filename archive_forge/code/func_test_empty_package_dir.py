import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import requires_subprocess
def test_empty_package_dir(self):
    sources = self.mkdtemp()
    open(os.path.join(sources, '__init__.py'), 'w').close()
    testdir = os.path.join(sources, 'doc')
    os.mkdir(testdir)
    open(os.path.join(testdir, 'testfile'), 'w').close()
    os.chdir(sources)
    dist = Distribution({'packages': ['pkg'], 'package_dir': {'pkg': ''}, 'package_data': {'pkg': ['doc/*']}})
    dist.script_name = os.path.join(sources, 'setup.py')
    dist.script_args = ['build']
    dist.parse_command_line()
    try:
        dist.run_commands()
    except DistutilsFileError:
        self.fail("failed package_data test when package_dir is ''")