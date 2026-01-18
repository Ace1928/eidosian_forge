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
def test_defaultTestResult(self):

    class Foo(unittest.TestCase):

        def runTest(self):
            pass
    result = Foo().defaultTestResult()
    self.assertEqual(type(result), unittest.TestResult)