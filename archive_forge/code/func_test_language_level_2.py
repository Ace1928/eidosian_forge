from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_language_level_2(self):
    options, args = self.parse_args(['-2'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['language_level']))
    self.assertEqual(options.language_level, 2)