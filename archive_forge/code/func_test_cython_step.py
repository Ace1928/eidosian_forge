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
def test_cython_step(self):
    gdb.execute('cy break codefile.spam')
    gdb.execute('run', to_string=True)
    self.lineno_equals('def spam(a=0):')
    gdb.execute('cy step', to_string=True)
    self.lineno_equals('b = c = d = 0')
    self.command = 'cy step'
    self.step([('b', 0)], source_line='b = 1')
    self.step([('b', 1), ('c', 0)], source_line='c = 2')
    self.step([('c', 2)], source_line='int(10)')
    self.step([], source_line='puts("spam")')
    gdb.execute('cont', to_string=True)
    self.assertEqual(len(gdb.inferiors()), 1)
    self.assertEqual(gdb.inferiors()[0].pid, 0)