import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
def test_check_ret(self):
    with self.assertRaises(self._exception_type):
        guard_ret(float(1.2))