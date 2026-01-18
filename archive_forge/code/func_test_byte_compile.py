import sys
import os
import importlib.util
import unittest
from distutils.command.install_lib import install_lib
from distutils.extension import Extension
from distutils.tests import support
from distutils.errors import DistutilsOptionError
from test.support import requires_subprocess
@unittest.skipIf(sys.dont_write_bytecode, 'byte-compile disabled')
@requires_subprocess()
def test_byte_compile(self):
    project_dir, dist = self.create_dist()
    os.chdir(project_dir)
    cmd = install_lib(dist)
    cmd.compile = cmd.optimize = 1
    f = os.path.join(project_dir, 'foo.py')
    self.write_file(f, '# python file')
    cmd.byte_compile([f])
    pyc_file = importlib.util.cache_from_source('foo.py', optimization='')
    pyc_opt_file = importlib.util.cache_from_source('foo.py', optimization=cmd.optimize)
    self.assertTrue(os.path.exists(pyc_file))
    self.assertTrue(os.path.exists(pyc_opt_file))