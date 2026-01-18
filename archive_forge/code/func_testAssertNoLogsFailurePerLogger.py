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
def testAssertNoLogsFailurePerLogger(self):
    with self.assertRaises(self.failureException) as cm:
        with self.assertLogs(log_quux):
            with self.assertNoLogs(logger=log_foo):
                log_quux.error('1')
                log_foobar.info('2')
    self.assertEqual(str(cm.exception), "Unexpected logs found: ['INFO:foo.bar:2']")