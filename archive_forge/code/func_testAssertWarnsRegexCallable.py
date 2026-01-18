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
def testAssertWarnsRegexCallable(self):

    def _runtime_warn(msg):
        warnings.warn(msg, RuntimeWarning)
    self.assertWarnsRegex(RuntimeWarning, 'o+', _runtime_warn, 'foox')
    with self.assertRaises(self.failureException):
        self.assertWarnsRegex(RuntimeWarning, 'o+', lambda: 0)
    with self.assertRaises(TypeError):
        self.assertWarnsRegex(RuntimeWarning, 'o+', None)
    with warnings.catch_warnings():
        warnings.simplefilter('default', RuntimeWarning)
        with self.assertRaises(self.failureException):
            self.assertWarnsRegex(DeprecationWarning, 'o+', _runtime_warn, 'foox')
    with self.assertRaises(self.failureException):
        self.assertWarnsRegex(RuntimeWarning, 'o+', _runtime_warn, 'barz')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with self.assertRaises((RuntimeWarning, self.failureException)):
            self.assertWarnsRegex(RuntimeWarning, 'o+', _runtime_warn, 'barz')