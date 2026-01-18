from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_directive_multiple_v2(self):
    options, args = self.parse_args(['-X', 'cdivision=True,c_string_type=bytes'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['directives']))
    self.assertEqual(options.directives['cdivision'], True)
    self.assertEqual(options.directives['c_string_type'], 'bytes')