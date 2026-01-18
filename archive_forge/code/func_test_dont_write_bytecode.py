import sys
import os
import importlib.util
import unittest
from distutils.command.install_lib import install_lib
from distutils.extension import Extension
from distutils.tests import support
from distutils.errors import DistutilsOptionError
from test.support import requires_subprocess
@requires_subprocess()
def test_dont_write_bytecode(self):
    dist = self.create_dist()[1]
    cmd = install_lib(dist)
    cmd.compile = 1
    cmd.optimize = 1
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        cmd.byte_compile([])
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    self.assertIn('byte-compiling is disabled', self.logs[0][1] % self.logs[0][2])