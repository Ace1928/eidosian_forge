from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_build_long(self):
    options, args = self.parse_args(['--build'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['build']))
    self.assertEqual(options.build, True)