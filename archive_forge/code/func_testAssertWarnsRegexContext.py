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
def testAssertWarnsRegexContext(self):

    def _runtime_warn(msg):
        warnings.warn(msg, RuntimeWarning)
    _runtime_warn_lineno = inspect.getsourcelines(_runtime_warn)[1]
    with self.assertWarnsRegex(RuntimeWarning, 'o+') as cm:
        _runtime_warn('foox')
    self.assertIsInstance(cm.warning, RuntimeWarning)
    self.assertEqual(cm.warning.args[0], 'foox')
    self.assertIn('test_case.py', cm.filename)
    self.assertEqual(cm.lineno, _runtime_warn_lineno + 1)
    with self.assertRaises(self.failureException):
        with self.assertWarnsRegex(RuntimeWarning, 'o+'):
            pass
    with self.assertRaisesRegex(self.failureException, 'foobar'):
        with self.assertWarnsRegex(RuntimeWarning, 'o+', msg='foobar'):
            pass
    with self.assertRaisesRegex(TypeError, 'foobar'):
        with self.assertWarnsRegex(RuntimeWarning, 'o+', foobar=42):
            pass
    with warnings.catch_warnings():
        warnings.simplefilter('default', RuntimeWarning)
        with self.assertRaises(self.failureException):
            with self.assertWarnsRegex(DeprecationWarning, 'o+'):
                _runtime_warn('foox')
    with self.assertRaises(self.failureException):
        with self.assertWarnsRegex(RuntimeWarning, 'o+'):
            _runtime_warn('barz')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with self.assertRaises((RuntimeWarning, self.failureException)):
            with self.assertWarnsRegex(RuntimeWarning, 'o+'):
                _runtime_warn('barz')