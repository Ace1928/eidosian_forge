import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_make_file(self):
    cmd = self.cmd
    self.assertRaises(TypeError, cmd.make_file, infiles=1, outfile='', func='func', args=())

    def _execute(func, args, exec_msg, level):
        self.assertEqual(exec_msg, 'generating out from in')
    cmd.force = True
    cmd.execute = _execute
    cmd.make_file(infiles='in', outfile='out', func='func', args=())