from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
def test_cython_next(self):
    self.break_and_run('c = 2')
    lines = ('int(10)', 'puts("spam")', 'os.path.join("foo", "bar")', 'some_c_function()')
    for line in lines:
        gdb.execute('cy next')
        self.lineno_equals(line)