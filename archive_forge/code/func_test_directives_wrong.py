from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_directives_wrong(self):
    directives = {'auto_pickle': 42, 'auto_pickle': 'NONONO', 'c_string_type': 'bites'}
    for key, value in directives.items():
        cmd = '{key}={value}'.format(key=key, value=str(value))
        with self.assertRaises(ValueError, msg='Error for option: ' + cmd) as context:
            options, args = self.parse_args(['-X', cmd])