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
def testTruncateMessage(self):
    self.maxDiff = 1
    message = self._truncateMessage('foo', 'bar')
    omitted = unittest.case.DIFF_OMITTED % len('bar')
    self.assertEqual(message, 'foo' + omitted)
    self.maxDiff = None
    message = self._truncateMessage('foo', 'bar')
    self.assertEqual(message, 'foobar')
    self.maxDiff = 4
    message = self._truncateMessage('foo', 'bar')
    self.assertEqual(message, 'foobar')