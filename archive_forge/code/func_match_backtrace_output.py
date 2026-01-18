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
def match_backtrace_output(result):
    assert re.search('\\#\\d+ *0x.* in spam\\(\\) at .*codefile\\.pyx:22', result), result
    assert 'os.path.join("foo", "bar")' in result, result