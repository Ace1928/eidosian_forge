from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_directive_no_value(self):
    with self.assertRaises(ValueError) as context:
        options, args = self.parse_args(['-X', 'cdivision'])