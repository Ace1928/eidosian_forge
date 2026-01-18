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
def testAssertLogsFailureMismatchingLogger(self):
    with self.assertLogs('quux', level='ERROR'):
        with self.assertRaises(self.failureException):
            with self.assertLogs('foo'):
                log_quux.error('1')