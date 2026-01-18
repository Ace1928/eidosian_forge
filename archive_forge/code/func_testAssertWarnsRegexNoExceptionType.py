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
def testAssertWarnsRegexNoExceptionType(self):
    with self.assertRaises(TypeError):
        self.assertWarnsRegex()
    with self.assertRaises(TypeError):
        self.assertWarnsRegex(UserWarning)
    with self.assertRaises(TypeError):
        self.assertWarnsRegex(1, 'expect')
    with self.assertRaises(TypeError):
        self.assertWarnsRegex(object, 'expect')
    with self.assertRaises(TypeError):
        self.assertWarnsRegex((UserWarning, 1), 'expect')
    with self.assertRaises(TypeError):
        self.assertWarnsRegex((UserWarning, object), 'expect')
    with self.assertRaises(TypeError):
        self.assertWarnsRegex((UserWarning, Exception), 'expect')