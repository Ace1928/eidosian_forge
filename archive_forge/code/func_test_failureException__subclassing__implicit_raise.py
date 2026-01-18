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
def test_failureException__subclassing__implicit_raise(self):
    events = []
    result = LoggingResult(events)

    class Foo(unittest.TestCase):

        def test(self):
            self.fail('foo')
        failureException = RuntimeError
    self.assertIs(Foo('test').failureException, RuntimeError)
    Foo('test').run(result)
    expected = ['startTest', 'addFailure', 'stopTest']
    self.assertEqual(events, expected)