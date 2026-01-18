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
def testAssertNotRaisesRegex(self):
    self.assertRaisesRegex(self.failureException, '^Exception not raised by <lambda>$', self.assertRaisesRegex, Exception, re.compile('x'), lambda: None)
    self.assertRaisesRegex(self.failureException, '^Exception not raised by <lambda>$', self.assertRaisesRegex, Exception, 'x', lambda: None)
    with self.assertRaisesRegex(self.failureException, 'foobar'):
        with self.assertRaisesRegex(Exception, 'expect', msg='foobar'):
            pass
    with self.assertRaisesRegex(TypeError, 'foobar'):
        with self.assertRaisesRegex(Exception, 'expect', foobar=42):
            pass