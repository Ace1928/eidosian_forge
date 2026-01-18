from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_keep_going_short(self):
    options, args = self.parse_args(['-k'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['keep_going']))
    self.assertEqual(options.keep_going, True)