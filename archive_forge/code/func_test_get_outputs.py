import sys
import os
import importlib.util
import unittest
from distutils.command.install_lib import install_lib
from distutils.extension import Extension
from distutils.tests import support
from distutils.errors import DistutilsOptionError
from test.support import requires_subprocess
def test_get_outputs(self):
    project_dir, dist = self.create_dist()
    os.chdir(project_dir)
    os.mkdir('spam')
    cmd = install_lib(dist)
    cmd.compile = cmd.optimize = 1
    cmd.install_dir = self.mkdtemp()
    f = os.path.join(project_dir, 'spam', '__init__.py')
    self.write_file(f, '# python package')
    cmd.distribution.ext_modules = [Extension('foo', ['xxx'])]
    cmd.distribution.packages = ['spam']
    cmd.distribution.script_name = 'setup.py'
    outputs = cmd.get_outputs()
    self.assertEqual(len(outputs), 4, outputs)