import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def assert_line_location(self, expected, offset_from_caller):
    grandparent_co = self._get_grandparent_caller_code()
    lno = grandparent_co.co_firstlineno
    self.assertEqual(expected, lno + offset_from_caller)