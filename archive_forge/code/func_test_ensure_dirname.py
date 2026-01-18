import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_ensure_dirname(self):
    cmd = self.cmd
    cmd.option1 = os.path.dirname(__file__) or os.curdir
    cmd.ensure_dirname('option1')
    cmd.option2 = 'xxx'
    self.assertRaises(DistutilsOptionError, cmd.ensure_dirname, 'option2')