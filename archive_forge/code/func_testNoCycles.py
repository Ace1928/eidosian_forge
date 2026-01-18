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
@support.cpython_only
def testNoCycles(self):
    case = unittest.TestCase()
    wr = weakref.ref(case)
    with support.disable_gc():
        del case
        self.assertFalse(wr())