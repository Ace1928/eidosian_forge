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
def lineno_equals(self, source_line=None, lineno=None):
    if source_line is not None:
        lineno = test_libcython.source_to_lineno[source_line]
    frame = gdb.selected_frame()
    self.assertEqual(libcython.cython_info.lineno(frame), lineno)