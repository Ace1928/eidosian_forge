import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
def guard_ret(val) -> int:
    return val