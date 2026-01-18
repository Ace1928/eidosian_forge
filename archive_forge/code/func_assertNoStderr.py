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
@contextlib.contextmanager
def assertNoStderr(self):
    with captured_stderr() as buf:
        yield
    self.assertEqual(buf.getvalue(), '')