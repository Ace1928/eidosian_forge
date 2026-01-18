import unittest
from numba.tests.support import TestCase, run_in_subprocess
def test_no_accidental_warnings(self):
    code = 'import numba'
    flags = ['-Werror', '-Wignore::DeprecationWarning:packaging.version:']
    run_in_subprocess(code, flags)