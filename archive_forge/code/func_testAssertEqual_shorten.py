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
def testAssertEqual_shorten(self):
    old_threshold = self._diffThreshold
    self._diffThreshold = 0
    self.addCleanup(lambda: setattr(self, '_diffThreshold', old_threshold))
    s = 'x' * 100
    s1, s2 = (s + 'a', s + 'b')
    with self.assertRaises(self.failureException) as cm:
        self.assertEqual(s1, s2)
    c = 'xxxx[35 chars]' + 'x' * 61
    self.assertEqual(str(cm.exception), "'%sa' != '%sb'" % (c, c))
    self.assertEqual(s + 'a', s + 'a')
    p = 'y' * 50
    s1, s2 = (s + 'a' + p, s + 'b' + p)
    with self.assertRaises(self.failureException) as cm:
        self.assertEqual(s1, s2)
    c = 'xxxx[85 chars]xxxxxxxxxxx'
    self.assertEqual(str(cm.exception), "'%sa%s' != '%sb%s'" % (c, p, c, p))
    p = 'y' * 100
    s1, s2 = (s + 'a' + p, s + 'b' + p)
    with self.assertRaises(self.failureException) as cm:
        self.assertEqual(s1, s2)
    c = 'xxxx[91 chars]xxxxx'
    d = 'y' * 40 + '[56 chars]yyyy'
    self.assertEqual(str(cm.exception), "'%sa%s' != '%sb%s'" % (c, d, c, d))