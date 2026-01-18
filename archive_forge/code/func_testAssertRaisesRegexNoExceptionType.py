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
def testAssertRaisesRegexNoExceptionType(self):
    with self.assertRaises(TypeError):
        self.assertRaisesRegex()
    with self.assertRaises(TypeError):
        self.assertRaisesRegex(ValueError)
    with self.assertRaises(TypeError):
        self.assertRaisesRegex(1, 'expect')
    with self.assertRaises(TypeError):
        self.assertRaisesRegex(object, 'expect')
    with self.assertRaises(TypeError):
        self.assertRaisesRegex((ValueError, 1), 'expect')
    with self.assertRaises(TypeError):
        self.assertRaisesRegex((ValueError, object), 'expect')