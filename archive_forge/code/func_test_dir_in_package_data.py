import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import requires_subprocess
def test_dir_in_package_data(self):
    """
        A directory in package_data should not be added to the filelist.
        """
    sources = self.mkdtemp()
    pkg_dir = os.path.join(sources, 'pkg')
    os.mkdir(pkg_dir)
    open(os.path.join(pkg_dir, '__init__.py'), 'w').close()
    docdir = os.path.join(pkg_dir, 'doc')
    os.mkdir(docdir)
    open(os.path.join(docdir, 'testfile'), 'w').close()
    os.mkdir(os.path.join(docdir, 'otherdir'))
    os.chdir(sources)
    dist = Distribution({'packages': ['pkg'], 'package_data': {'pkg': ['doc/*']}})
    dist.script_name = os.path.join(sources, 'setup.py')
    dist.script_args = ['build']
    dist.parse_command_line()
    try:
        dist.run_commands()
    except DistutilsFileError:
        self.fail('failed package_data when data dir includes a dir')