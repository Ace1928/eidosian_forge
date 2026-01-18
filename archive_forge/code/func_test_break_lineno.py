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
def test_break_lineno(self):
    beginline = 'import os'
    nextline = 'cdef int c_var = 12'
    self.break_and_run(beginline)
    self.lineno_equals(beginline)
    step_result = gdb.execute('cy step', to_string=True)
    self.lineno_equals(nextline)
    assert step_result.rstrip().endswith(nextline)