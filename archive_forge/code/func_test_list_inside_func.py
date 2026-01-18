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
def test_list_inside_func(self):
    self.break_and_run('c = 2')
    result = gdb.execute('cy list', to_string=True)
    result = '\n'.join([line.rstrip() for line in result.split('\n')])
    self.assertEqual(correct_result_test_list_inside_func, result)