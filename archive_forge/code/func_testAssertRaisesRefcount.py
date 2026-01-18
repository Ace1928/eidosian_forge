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
def testAssertRaisesRefcount(self):

    def func():
        try:
            raise ValueError
        except ValueError:
            raise ValueError
    refcount = sys.getrefcount(func)
    self.assertRaises(ValueError, func)
    self.assertEqual(refcount, sys.getrefcount(func))