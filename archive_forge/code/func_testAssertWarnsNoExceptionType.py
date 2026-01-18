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
def testAssertWarnsNoExceptionType(self):
    with self.assertRaises(TypeError):
        self.assertWarns()
    with self.assertRaises(TypeError):
        self.assertWarns(1)
    with self.assertRaises(TypeError):
        self.assertWarns(object)
    with self.assertRaises(TypeError):
        self.assertWarns((UserWarning, 1))
    with self.assertRaises(TypeError):
        self.assertWarns((UserWarning, object))
    with self.assertRaises(TypeError):
        self.assertWarns((UserWarning, Exception))