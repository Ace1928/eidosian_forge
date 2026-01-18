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
def testAssertLogsFailureLevelTooHigh_FilterInRootLogger(self):
    with self.assertNoStderr():
        oldLevel = log_foo.level
        log_foo.setLevel(logging.INFO)
        try:
            with self.assertRaises(self.failureException):
                with self.assertLogs(level='WARNING'):
                    log_foo.info('1')
        finally:
            log_foo.setLevel(oldLevel)