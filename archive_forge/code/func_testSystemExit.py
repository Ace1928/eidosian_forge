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
def testSystemExit(self):

    def _raise(self=None):
        raise SystemExit

    def nothing(self):
        pass

    class Test1(unittest.TestCase):
        test_something = _raise

    class Test2(unittest.TestCase):
        setUp = _raise
        test_something = nothing

    class Test3(unittest.TestCase):
        test_something = nothing
        tearDown = _raise

    class Test4(unittest.TestCase):

        def test_something(self):
            self.addCleanup(_raise)
    for klass in (Test1, Test2, Test3, Test4):
        result = unittest.TestResult()
        klass('test_something').run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.testsRun, 1)