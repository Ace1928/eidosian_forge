import sys
import os
import importlib.util
import unittest
from distutils.command.install_lib import install_lib
from distutils.extension import Extension
from distutils.tests import support
from distutils.errors import DistutilsOptionError
from test.support import requires_subprocess
def test_finalize_options(self):
    dist = self.create_dist()[1]
    cmd = install_lib(dist)
    cmd.finalize_options()
    self.assertEqual(cmd.compile, 1)
    self.assertEqual(cmd.optimize, 0)
    cmd.optimize = 'foo'
    self.assertRaises(DistutilsOptionError, cmd.finalize_options)
    cmd.optimize = '4'
    self.assertRaises(DistutilsOptionError, cmd.finalize_options)
    cmd.optimize = '2'
    cmd.finalize_options()
    self.assertEqual(cmd.optimize, 2)