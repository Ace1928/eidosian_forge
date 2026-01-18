import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
def test_check_does_not_work_with_inner_func(self):

    def guard(val: int) -> int:
        return
    guard(float(1.2))