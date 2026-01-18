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
def testAssertDictEqualTruncates(self):
    test = unittest.TestCase('assertEqual')

    def truncate(msg, diff):
        return 'foo'
    test._truncateMessage = truncate
    try:
        test.assertDictEqual({}, {1: 0})
    except self.failureException as e:
        self.assertEqual(str(e), 'foo')
    else:
        self.fail('assertDictEqual did not fail')