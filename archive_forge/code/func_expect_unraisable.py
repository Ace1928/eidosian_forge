from ctypes import *
import contextlib
from test import support
import unittest
import sys
@contextlib.contextmanager
def expect_unraisable(self, exc_type, exc_msg=None):
    with support.catch_unraisable_exception() as cm:
        yield
        self.assertIsInstance(cm.unraisable.exc_value, exc_type)
        if exc_msg is not None:
            self.assertEqual(str(cm.unraisable.exc_value), exc_msg)
        self.assertEqual(cm.unraisable.err_msg, 'Exception ignored on calling ctypes callback function')
        self.assertIs(cm.unraisable.object, callback_func)