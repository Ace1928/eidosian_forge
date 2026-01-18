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
def test_cy_eval(self):
    self.break_and_run('os.path.join("foo", "bar")')
    result = gdb.execute('print $cy_eval("None")', to_string=True)
    assert re.match('\\$\\d+ = None\\n', result), result
    result = gdb.execute('print $cy_eval("[a]")', to_string=True)
    assert re.match('\\$\\d+ = \\[0\\]', result), result