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
def test_inner(self):
    self.break_and_run_func('inner')
    self.assertEqual('', gdb.execute('cy locals', to_string=True))
    gdb.execute('cy step')
    self.assertEqual(str(self.read_var('a')), "'an object'")
    print_result = gdb.execute('cy print a', to_string=True).strip()
    self.assertEqual(print_result, "a = 'an object'")