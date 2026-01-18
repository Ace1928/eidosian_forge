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
def testAssertNoLogsFailureFoundLogs(self):
    with self.assertRaises(self.failureException) as cm:
        with self.assertNoLogs():
            log_quux.error('1')
            log_foo.error('foo')
    self.assertEqual(str(cm.exception), "Unexpected logs found: ['ERROR:quux:1', 'ERROR:foo:foo']")