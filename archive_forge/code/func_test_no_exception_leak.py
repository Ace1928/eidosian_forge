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
def test_no_exception_leak(self):

    class MyException(Exception):
        ninstance = 0

        def __init__(self):
            MyException.ninstance += 1
            Exception.__init__(self)

        def __del__(self):
            MyException.ninstance -= 1

    class TestCase(unittest.TestCase):

        def test1(self):
            raise MyException()

        @unittest.expectedFailure
        def test2(self):
            raise MyException()
    for method_name in ('test1', 'test2'):
        testcase = TestCase(method_name)
        testcase.run()
        gc_collect()
        self.assertEqual(MyException.ninstance, 0)