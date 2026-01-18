from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_force_long(self):
    options, args = self.parse_args(['--force'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['force']))
    self.assertEqual(options.force, True)