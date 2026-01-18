from ctypes import *
import contextlib
from test import support
import unittest
import sys
def test_FloatDivisionError(self):
    cb = CFUNCTYPE(c_int, c_double)(callback_func)
    with self.expect_unraisable(ZeroDivisionError):
        cb(0.0)