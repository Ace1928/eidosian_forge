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
def testAssertWarnsContext(self):

    def _runtime_warn():
        warnings.warn('foo', RuntimeWarning)
    _runtime_warn_lineno = inspect.getsourcelines(_runtime_warn)[1]
    with self.assertWarns(RuntimeWarning) as cm:
        _runtime_warn()
    with self.assertWarns((DeprecationWarning, RuntimeWarning)) as cm:
        _runtime_warn()
    self.assertIsInstance(cm.warning, RuntimeWarning)
    self.assertEqual(cm.warning.args[0], 'foo')
    self.assertIn('test_case.py', cm.filename)
    self.assertEqual(cm.lineno, _runtime_warn_lineno + 1)
    with self.assertWarns(RuntimeWarning):
        _runtime_warn()
        _runtime_warn()
    with self.assertWarns(RuntimeWarning):
        warnings.warn('foo', category=RuntimeWarning)
    with self.assertRaises(self.failureException):
        with self.assertWarns(RuntimeWarning):
            pass
    with self.assertRaisesRegex(self.failureException, 'foobar'):
        with self.assertWarns(RuntimeWarning, msg='foobar'):
            pass
    with self.assertRaisesRegex(TypeError, 'foobar'):
        with self.assertWarns(RuntimeWarning, foobar=42):
            pass
    with warnings.catch_warnings():
        warnings.simplefilter('default', RuntimeWarning)
        with self.assertRaises(self.failureException):
            with self.assertWarns(DeprecationWarning):
                _runtime_warn()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with self.assertRaises(RuntimeWarning):
            with self.assertWarns(DeprecationWarning):
                _runtime_warn()