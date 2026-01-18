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
def testAssertNoLogsDefault(self):
    with self.assertRaises(self.failureException) as cm:
        with self.assertNoLogs():
            log_foo.info('1')
            log_foobar.debug('2')
    self.assertEqual(str(cm.exception), "Unexpected logs found: ['INFO:foo:1']")