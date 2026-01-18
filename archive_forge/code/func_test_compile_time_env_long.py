from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_compile_time_env_long(self):
    options, args = self.parse_args(['--compile-time-env', 'MYSIZE=10'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['compile_time_env']))
    self.assertEqual(options.compile_time_env['MYSIZE'], 10)