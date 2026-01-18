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
def testAddTypeEqualityFunc(self):

    class SadSnake(object):
        """Dummy class for test_addTypeEqualityFunc."""
    s1, s2 = (SadSnake(), SadSnake())
    self.assertFalse(s1 == s2)

    def AllSnakesCreatedEqual(a, b, msg=None):
        return type(a) == type(b) == SadSnake
    self.addTypeEqualityFunc(SadSnake, AllSnakesCreatedEqual)
    self.assertEqual(s1, s2)