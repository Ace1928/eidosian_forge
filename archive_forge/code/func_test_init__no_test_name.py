import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def test_init__no_test_name(self):

    class Test(unittest.TestCase):

        def runTest(self):
            raise MyException()

        def test(self):
            pass
    self.assertEqual(Test().id()[-13:], '.Test.runTest')
    test = unittest.TestCase()
    test.assertEqual(3, 3)
    with test.assertRaises(test.failureException):
        test.assertEqual(3, 2)
    with self.assertRaises(AttributeError):
        test.run()