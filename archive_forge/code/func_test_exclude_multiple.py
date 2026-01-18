from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_exclude_multiple(self):
    options, args = self.parse_args(['--exclude', '*.pyx', '--exclude', '*.py'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['excludes']))
    self.assertEqual(options.excludes, ['*.pyx', '*.py'])