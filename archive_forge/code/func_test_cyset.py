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
def test_cyset(self):
    self.break_and_run('os.path.join("foo", "bar")')
    gdb.execute('cy set a = $cy_eval("{None: []}")')
    stringvalue = self.read_var('a', cast_to=str)
    self.assertEqual(stringvalue, '{None: []}')