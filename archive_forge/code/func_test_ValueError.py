from ctypes import *
import contextlib
from test import support
import unittest
import sys
def test_ValueError(self):
    cb = CFUNCTYPE(c_int, c_int)(callback_func)
    with self.expect_unraisable(ValueError, '42'):
        cb(42)